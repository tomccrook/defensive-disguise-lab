"""Frozen game-level research on hash-verified historical NFL tracking."""

import copy
import json
import subprocess
import time
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import torch
from scipy.special import logsumexp
from torch_geometric.data import Data

from disguise_lab.audit import file_hash
from disguise_lab.columnar import opponent_split
from disguise_lab.contracts import digest
from disguise_lab.features import Features
from disguise_lab.models import (
    Baseline,
    Family,
    MovementModel,
    SharedMixtureModel,
    masked_loss,
    matched_width,
    mixture_loss,
    parameter_count,
)
from disguise_lab.probability import (
    gaussian_nll,
    gaussian_samples,
)
from disguise_lab.registry import Registry
from disguise_lab.study import FAMILIES, write_json


class CachedNFL:
    def __init__(self, root: Path):
        self.root = root
        self.dataset = json.loads((root / "dataset.json").read_text())
        self.metadata = json.loads((root / "plays.json").read_text())
        self.history = np.load(root / "history.npy", mmap_mode="r")
        self.anchor = np.load(root / "anchor.npy", mmap_mode="r")
        self.last = np.load(root / "last.npy", mmap_mode="r")
        self.velocity = np.load(root / "velocity.npy", mmap_mode="r")
        self.y = np.load(root / "target.npy", mmap_mode="r")
        self.mask = np.load(root / "mask.npy", mmap_mode="r")
        self.defenders = np.arange(22) >= 11
        self.edge_index = np.array(np.where(~np.eye(22, dtype=bool)))
        own = self.history[:, :, :, -1, :]
        pooled = np.concatenate(
            [
                part
                for side in [slice(0, 11), slice(11, 22)]
                for part in [own[:, :, side].mean(2), own[:, :, side].std(2)]
            ],
            axis=-1,
        )
        last = np.asarray(self.last)
        distances = np.linalg.norm(last[:, :, :, None, :] - last[:, :, None, :, :], axis=-1)
        opposite = self.defenders[:, None] != self.defenders[None, :]
        nearest = np.min(np.where(opposite[None, None], distances, np.inf), axis=-1)
        self.geometry = np.concatenate(
            [
                own[:, :, 11:],
                np.broadcast_to(pooled[:, :, None], (*pooled.shape[:2], 11, pooled.shape[-1])),
                nearest[:, :, 11:, None] / 10,
            ],
            axis=-1,
        )

    def combinations(self, indices: np.ndarray) -> np.ndarray:
        return np.array(
            [(int(p), c, h) for p in indices for c in range(3) for h in range(2)], dtype=int
        )

    def features(self, p: int, c: int, h: int) -> Features:
        last = np.array(self.last[p, c])
        velocity = np.array(self.velocity[p, c])
        send, receive = self.edge_index
        delta = last[send] - last[receive]
        edges = np.column_stack(
            [
                delta / 10,
                np.linalg.norm(delta, axis=-1) / 10,
                (velocity[send] - velocity[receive]) / 5,
                self.defenders[send] == self.defenders[receive],
            ]
        )
        return Features(
            tuple(self.metadata[p]["entity_ids"]),
            np.array(self.history[p, c]),
            np.array(self.anchor[p]),
            last,
            velocity,
            self.defenders,
            self.edge_index,
            edges,
            np.tile([[-1, -0.5, -0.1][c], [0.5, 1][h]], (22, 1)),
        )

    def matrix(self, combos: np.ndarray, static: bool = False) -> np.ndarray:
        p, c, h = combos.T
        x = self.geometry[p, c].copy()
        if static:
            # Own velocity plus both team mean/std velocity features.
            x[:, :, [2, 3, 8, 9, 14, 15, 20, 21, 26, 27]] = 0
        context = np.stack([np.array([-1, -0.5, -0.1])[c], np.array([0.5, 1])[h]], axis=-1)
        return np.concatenate([x, np.broadcast_to(context[:, None], (len(x), 11, 2))], axis=-1)

    def batch(
        self,
        combos: np.ndarray,
        device: str = "cpu",
        noise: float = 0,
        reduced_history: bool = False,
    ) -> Data:
        p, c, h = combos.T
        n = len(p)

        def tensor(x, dtype=torch.float32):
            return torch.tensor(np.array(x), dtype=dtype, device=device)

        history = tensor(self.history[p, c]).reshape(n * 22, 21, 6)
        if noise:
            history = history.clone()
            history[:, :, :2] += torch.randn_like(history[:, :, :2]) * (noise / 10)
        if reduced_history:
            history = history[:, -6:]
        last = tensor(self.last[p, c]).reshape(-1, 2)
        vel = tensor(self.velocity[p, c]).reshape(-1, 2)
        send, receive = self.edge_index
        edges = tensor(self.edge_index[None] + np.arange(n)[:, None, None] * 22, torch.long)
        edges = edges.permute(1, 0, 2).reshape(2, -1)
        s, r = edges
        delta = last[s] - last[r]
        defender = tensor(np.tile(self.defenders, n), torch.bool)
        attr = torch.cat(
            [
                delta / 10,
                delta.norm(dim=-1, keepdim=True) / 10,
                (vel[s] - vel[r]) / 5,
                (defender[s] == defender[r])[:, None].to(last.dtype),
            ],
            dim=-1,
        )
        context = np.stack([np.array([-1, -0.5, -0.1])[c], np.array([0.5, 1])[h]], axis=-1)
        return Data(
            x=history,
            edge_index=edges,
            edge_attr=attr,
            defender=defender,
            anchor=tensor(self.anchor[p]).reshape(-1, 2),
            last=last,
            velocity=vel,
            context=tensor(np.repeat(context, 22, axis=0)),
            batch=tensor(np.repeat(np.arange(n), 22), torch.long),
            y=tensor(self.y[p, h]).reshape(-1, 2),
            target_mask=tensor(self.mask[p, h], torch.bool).reshape(-1),
            num_nodes=n * 22,
        )


def freeze_splits(data: CachedNFL, config: dict, out: Path) -> dict:
    mapping = {
        m["game_id"]: next(
            s for s, weeks in config["chronological_weeks"].items() if m["week"] in weeks
        )
        for m in data.metadata
    }
    chrono = {
        "protocol": "chronological-whole-games",
        "games": mapping,
        "synthetic": False,
        "weeks": config["chronological_weeks"],
        "final_nfl_test_exposed": False,
    }
    chrono["split_id"] = digest(chrono)
    opponent = opponent_split(
        data.metadata, set(config["opponent_test"]), set(config["opponent_validation"])
    )
    for split in [chrono, opponent]:
        for s, min_count in config["minimum_games"].items():
            if sum(v == s for v in split["games"].values()) < min_count:
                raise ValueError(f"insufficient independent {s} games")
    splits = {"chronological": chrono, "opponent": opponent}
    counts = {
        protocol: {
            s: {
                "games": sum(v == s for v in split["games"].values()),
                "plays": sum(split["games"][m["game_id"]] == s for m in data.metadata),
            }
            for s in ["train", "validation", "test"]
        }
        for protocol, split in splits.items()
    }
    value = {
        "config": config,
        "splits": splits,
        "counts": counts,
        "dataset_id": data.dataset["dataset_id"],
        "dataset_hash": file_hash(data.root / "dataset.json"),
        "protected_test_policy": (
            "No model-dependent test inspection before freeze of all protocols/models."
        ),
    }
    target = out / "preregistration.json"
    if target.exists() and json.loads(target.read_text()) != value:
        raise ValueError("preregistered study cannot be overwritten")
    write_json(target, value)
    return splits


def neural_fit(
    data: CachedNFL,
    train: np.ndarray,
    val: np.ndarray,
    variant: str,
    config: dict,
    seed: int,
    device: str,
):
    torch.manual_seed(seed)
    family = cast(
        Family,
        "pooled" if "pooled" in variant else "temporal" if "temporal" in variant else "graph",
    )
    mixture = "mixture" in variant
    width = matched_width() if family == "pooled" else 24
    if mixture and family == "pooled":
        capacity = parameter_count(SharedMixtureModel("graph", 24))
        width = min(
            range(8, 97),
            key=lambda w: abs(parameter_count(SharedMixtureModel("pooled", w)) - capacity),
        )
    model = (
        SharedMixtureModel(family, width)
        if mixture
        else MovementModel(
            family,
            width,
            static_history=variant == "temporal_static",
            no_side=variant == "graph_no_side",
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    rng = np.random.default_rng(seed)
    best = float("inf")
    best_state = None
    history = []
    patience = 0
    start = time.perf_counter()
    budget = config["budget"]
    for epoch in range(budget["epochs"]):
        model.train()
        total = 0.0
        count = 0
        for ids in np.array_split(
            rng.permutation(len(train)), max(1, int(np.ceil(len(train) / budget["batch_size"])))
        ):
            batch = data.batch(train[ids], device)
            optimizer.zero_grad()
            result = model(batch)
            loss = (
                mixture_loss(
                    result[0],
                    result[1],
                    result[2],
                    batch.y,
                    batch.target_mask,
                    batch.batch[batch.defender],
                )
                if mixture
                else masked_loss(result[0], result[1], batch.y, batch.target_mask)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(ids)
            count += len(ids)
        model.eval()
        validation = 0.0
        n = 0
        with torch.no_grad():
            for offset in range(0, len(val), budget["batch_size"]):
                batch = data.batch(val[offset : offset + budget["batch_size"]], device)
                result = model(batch)
                loss = (
                    mixture_loss(
                        result[0],
                        result[1],
                        result[2],
                        batch.y,
                        batch.target_mask,
                        batch.batch[batch.defender],
                    )
                    if mixture
                    else masked_loss(result[0], result[1], batch.y, batch.target_mask)
                )
                size = len(val[offset : offset + budget["batch_size"]])
                validation += float(loss.cpu()) * size
                n += size
        metric = validation / n
        history.append(
            {
                "epoch": epoch + 1,
                "train_nll": total / count,
                "validation_nll": metric,
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        print(json.dumps({"variant": variant, **history[-1]}), flush=True)
        if metric < best - 1e-4:
            best = metric
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= 6 or time.perf_counter() - start >= budget["seconds_per_neural_trial"]:
            break
    if best_state is None:
        raise RuntimeError("no valid validation checkpoint")
    model.load_state_dict(best_state)
    model = model.cpu()
    model.eval()
    return model, history


def distributions(
    data: CachedNFL,
    model,
    combos: np.ndarray,
    device: str = "cpu",
    noise: float = 0,
    reduced_history: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if isinstance(model, Baseline):
        x = data.matrix(combos, getattr(model, "static_geometry", False))
        p, c, h = combos.T
        if model.family == "geometry":
            mean = model.estimator.predict(x.reshape(-1, x.shape[-1])).reshape(-1, 11, 2)
            if getattr(model, "anchor_residual", False):
                mean += data.last[p, c, 11:] - data.anchor[p, 11:]
        else:
            mean = np.array(data.last[p, c, 11:] - data.anchor[p, 11:])
            if model.family == "velocity":
                mean += (
                    data.velocity[p, c, 11:]
                    * (np.array([0.5, 1])[h] - np.array([-1, -0.5, -0.1])[c])[:, None, None]
                )
        covariance = np.array(
            [model.context_covariance[(int(a), int(b))] for a, b in zip(c, h, strict=True)]
        )
        return mean, np.broadcast_to(covariance[:, None], (len(mean), 11, 2, 2)).copy(), None
    means, covs, weights = [], [], []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(combos), 128):
            batch = data.batch(combos[offset : offset + 128], device, noise, reduced_history)
            result = model(batch)
            mixture = len(result) == 3
            means.append(result[0].cpu().numpy().reshape(-1, 11, *(result[0].shape[1:])))
            covs.append(result[1].cpu().numpy().reshape(-1, 11, *(result[1].shape[1:])))
            if mixture:
                weights.append(result[2].exp().cpu().numpy())
    return np.concatenate(means), np.concatenate(covs), np.concatenate(weights) if weights else None


def marginal_nll(mean, cov, y, weights=None):
    if weights is None:
        return gaussian_nll(mean, cov, y)
    return -logsumexp(
        np.log(weights[:, None, :]) + (-gaussian_nll(mean, cov, y[:, :, None, :])), axis=-1
    )


def moments(mean, cov, weights=None):
    if weights is None:
        return mean, cov
    point = (weights[:, None, :, None] * mean).sum(2)
    delta = mean - point[:, :, None]
    variance = (
        weights[:, None, :, None, None] * (cov + delta[:, :, :, :, None] * delta[:, :, :, None, :])
    ).sum(2)
    return point, variance


def draws(mean, cov, count, seed, weights=None):
    if weights is None:
        return gaussian_samples(mean.reshape(-1, 2), cov.reshape(-1, 2, 2), count, seed)
    rng = np.random.default_rng(seed)
    p, n, k, _ = mean.shape
    # A single scene component is shared across all eleven defenders in each draw.
    component = (
        (rng.random((count, p, 1)) > np.cumsum(weights, axis=-1)[None]).sum(-1).clip(0, k - 1)
    )
    choice = np.broadcast_to(component[:, :, None, None, None], (count, p, n, 1, 2))
    chosen = np.take_along_axis(
        np.broadcast_to(mean[None], (count, *mean.shape)), choice, axis=3
    ).squeeze(3)
    cchoice = np.broadcast_to(component[:, :, None, None, None, None], (count, p, n, 1, 2, 2))
    variance = np.take_along_axis(
        np.broadcast_to(cov[None], (count, *cov.shape)), cchoice, axis=3
    ).squeeze(3)
    z = rng.normal(size=(count, p, n, 2))
    return (chosen + np.einsum("mpnij,mpnj->mpni", np.linalg.cholesky(variance), z)).reshape(
        count, p * n, 2
    )


def calibrate(data, combos, mean, cov, weights):
    from scipy.optimize import minimize_scalar

    p, c, h = combos.T
    y = data.y[p, h]
    mask = data.mask[p, h]
    scales = {}
    sharp = {}
    for a in range(3):
        for b in range(2):
            select = (c == a) & (h == b)
            observed = mask[select]
            if observed.sum() < 100:
                raise ValueError("insufficient validation targets for calibration")

            def objective(log_scale, select=select, observed=observed):
                values = marginal_nll(
                    mean[select],
                    cov[select] * np.exp(log_scale),
                    y[select],
                    None if weights is None else weights[select],
                )
                return float(values[observed].mean())

            result = minimize_scalar(objective, bounds=(-3, 4.6), method="bounded")
            scale = float(np.exp(result.x))
            scales[(a, b)] = scale
            _, variance = moments(
                mean[select], cov[select] * scale, None if weights is None else weights[select]
            )
            sharp[(a, b)] = float(
                np.quantile(np.trace(variance, axis1=-2, axis2=-1)[observed], 0.25)
            )
    return scales, sharp


def fit_protocol(
    data: CachedNFL,
    split: dict,
    protocol: str,
    config: dict,
    out: Path,
    registry: Registry,
    device: str,
) -> dict:
    groups = {
        s: np.array([i for i, m in enumerate(data.metadata) if split["games"][m["game_id"]] == s])
        for s in ["train", "validation", "test"]
    }
    train = data.combinations(groups["train"])
    val = data.combinations(groups["validation"])
    variants = list(FAMILIES) + (
        [
            "geometry_static",
            "temporal_static",
            "graph_no_side",
            "pooled_mixture",
            "graph_mixture",
            "pooled_seed1730",
            "graph_seed1730",
        ]
        if protocol == "chronological"
        else []
    )
    bundles = {}
    location = out / protocol
    location.mkdir(parents=True, exist_ok=True)
    code_hash = digest({str(p): p.read_text() for p in sorted(Path("src").rglob("*.py"))})
    for variant in variants:
        checkpoint = location / f"{variant}.joblib"
        if checkpoint.exists():
            completed = [
                r
                for r in registry.records()
                if r["status"] == "succeeded" and r["result"].get("checkpoint") == str(checkpoint)
            ]
            if not completed or (
                completed[-1]["result"]["checkpoint_sha256"] != file_hash(checkpoint)
                or completed[-1]["config"]["config_hash"] != digest(config)
                or completed[-1]["config"]["split_id"] != split["split_id"]
            ):
                raise ValueError("resume checkpoint does not match registered frozen study")
            bundles[variant] = joblib.load(checkpoint)
            continue
        seed = config["extra_seed"] if "seed1730" in variant else config["seed"]
        family = variant.split("_")[0]
        trial_config = {
            "variant": variant,
            "protocol": protocol,
            "seed": seed,
            "device": device,
            "dataset_id": data.dataset["dataset_id"],
            "split_id": split["split_id"],
            "config_hash": digest(config),
            "source_tree_hash": code_hash,
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "scope": "historical NFL 2022; league-wide; no test selection",
            "max_seconds": config["budget"]["seconds_per_neural_trial"],
        }
        trial = registry.start(trial_config)
        start = time.perf_counter()
        try:
            if family in FAMILIES[:3]:
                model = Baseline(
                    family, static_geometry=variant == "geometry_static", iterations=250
                )
                p, c, h = train.T
                y = data.y[p, h]
                mask = data.mask[p, h]
                if family == "geometry":
                    x = data.matrix(train, model.static_geometry)
                    model.estimator.fit(x[mask], y[mask])
                model.context_covariance = {(a, b): np.eye(2) for a in range(3) for b in range(2)}
                mean, _, _ = distributions(data, model, train)
                for a in range(3):
                    for b in range(2):
                        take = (c == a) & (h == b)
                        r = (y[take] - mean[take])[mask[take]]
                        model.context_covariance[(a, b)] = r.T @ r / len(r) + np.eye(2) * 0.15**2
                model.covariance = model.context_covariance[(2, 1)]
                history = []
            else:
                model, history = neural_fit(data, train, val, variant, config, seed, device)
            mean, cov, weights = distributions(data, model, val, device)
            scales, sharp = calibrate(data, val, mean, cov, weights)
            bundle = {
                "model": model.cpu() if isinstance(model, torch.nn.Module) else model,
                "scales": scales,
                "sharp_thresholds": sharp,
                "scale": scales[(2, 1)],
                "model_id": trial,
                "dataset_id": data.dataset["dataset_id"],
                "split_id": split["split_id"],
                "calibration_id": digest({str(k): v for k, v in scales.items()}),
                "variant": variant,
                "protocol": protocol,
            }
            joblib.dump(bundle, checkpoint)
            registry.finish(
                trial,
                "succeeded",
                {
                    "seconds": time.perf_counter() - start,
                    "parameters": parameter_count(model)
                    if isinstance(model, torch.nn.Module)
                    else None,
                    "history": history,
                    "calibration_scales": {str(k): v for k, v in scales.items()},
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": file_hash(checkpoint),
                    "completed_epochs": len(history),
                    "device": device,
                    "dtype": "float32",
                },
            )
            bundles[variant] = bundle
            print(
                f"completed {protocol}/{variant} in {time.perf_counter() - start:.1f}s", flush=True
            )
        except BaseException as error:
            registry.finish(
                trial,
                "cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                {
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "seconds": time.perf_counter() - start,
                },
            )
            raise
    return bundles
