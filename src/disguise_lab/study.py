"""Bounded synthetic end-to-end study; never presents fixture scores as NFL evidence."""

import copy
import json
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import polars as pl
import torch
from torch_geometric.data import Batch

from disguise_lab.audit import file_hash
from disguise_lab.columnar import summarize_parquet, write_partition
from disguise_lab.contracts import DefenderForecast, DefenderMode, Forecast, ForecastInput, digest
from disguise_lab.features import Features, build_features
from disguise_lab.fixtures import (
    DATASET,
    chronological_split,
    generate,
    input_from_rows,
    reveal_from_rows,
)
from disguise_lab.models import (
    Baseline,
    Family,
    MovementModel,
    SharedMixtureModel,
    masked_loss,
    matched_width,
    parameter_count,
)
from disguise_lab.probability import (
    belief_change,
    clustered_interval,
    coverage,
    energy_score,
    fit_scale,
    gaussian_nll,
    gaussian_samples,
    mahalanobis,
)
from disguise_lab.registry import Registry

FAMILIES: list[Family] = ["persistence", "velocity", "geometry", "temporal", "pooled", "graph"]


def write_json(path: Path, value: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def targets(reveal) -> tuple[np.ndarray, np.ndarray]:
    mask = np.array([o.displacement is not None for o in reveal.outcomes])
    return np.array([o.displacement or (0, 0) for o in reveal.outcomes]), mask


def predict(model, features: Features) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(model, Baseline):
        return model.predict(features)
    model.eval()
    with torch.no_grad():
        mean, cov = model(features.graph())
    return mean.numpy(), cov.numpy()


def train_neural(family, train, validation, width, epochs, seconds):
    torch.manual_seed(1729)
    model = MovementModel(family, width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)

    def graph(example):
        f, y, mask = example
        g = f.graph()
        # Full-node y/mask keeps PyG batching order aligned; offense masks remain false.
        g.y = torch.zeros(22, 2)
        g.y[f.defenders] = torch.tensor(y, dtype=torch.float32)
        g.target_mask = torch.zeros(22, dtype=torch.bool)
        g.target_mask[f.defenders] = torch.tensor(mask)
        return g

    train_graphs, val_graphs = [graph(e) for e in train], [graph(e) for e in validation]
    best, best_state, patience = float("inf"), None, 0
    history = []
    start = time.perf_counter()
    rng = np.random.default_rng(1729)
    for epoch in range(epochs):
        if time.perf_counter() - start > seconds:
            break
        model.train()
        indices = rng.permutation(len(train_graphs))
        losses = []
        for offset in range(0, len(indices), 32):
            batch = Batch.from_data_list([train_graphs[i] for i in indices[offset : offset + 32]])
            optimizer.zero_grad()
            mean, cov = model(batch)
            loss = masked_loss(
                mean, cov, batch.y[batch.defender], batch.target_mask[batch.defender]
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training NLL")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            sums, count = 0.0, 0
            for offset in range(0, len(val_graphs), 32):
                batch = Batch.from_data_list(val_graphs[offset : offset + 32])
                mean, cov = model(batch)
                mask = batch.target_mask[batch.defender]
                n = int(mask.sum())
                sums += float(masked_loss(mean, cov, batch.y[batch.defender], mask)) * n
                count += n
            val_loss = sums / count
        history.append(
            {"epoch": epoch + 1, "train_nll": float(np.mean(losses)), "validation_nll": val_loss}
        )
        if val_loss < best - 1e-4:
            best, best_state, patience = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
        if patience >= 7:
            break
    if best_state is None:
        raise RuntimeError("budget expired before first validation")
    model.load_state_dict(best_state)
    return model, history


def run_synthetic(root: Path, epochs: int = 20, games: int = 18, plays: int = 8) -> Path:
    if not 1 <= epochs <= 40 or not 6 <= games <= 30 or not 1 <= plays <= 16:
        raise ValueError("fixture request exceeds declared budget")
    torch.set_num_threads(4)
    config = json.loads(Path("configs/study.json").read_text())
    source_hash = digest({str(p): p.read_text() for p in sorted(Path("src").rglob("*.py"))})
    registry = Registry(root / "trials.sqlite", config["budget"]["max_trials"])
    run_id = f"synthetic-{uuid4().hex[:12]}"
    out = root / run_id
    out.mkdir(parents=True, exist_ok=False)
    rows, metadata = generate(games, plays)
    split = chronological_split(metadata)
    dataset_id = DATASET + "-" + digest({"games": games, "plays": plays})[:12]
    write_json(out / "split.json", split)
    partition_manifest = write_partition(rows, out / "tracking.parquet")
    write_json(
        out / "dataset.json",
        {
            "dataset_id": dataset_id,
            "synthetic": True,
            "season": None,
            "games": games,
            "plays": len(metadata),
            "rows": rows.height,
            "cadence_hz": 10,
            "generator_seed": 1729,
            "tracking_hash": digest(rows.to_dicts()),
            "partition": partition_manifest,
            "duckdb_summary": summarize_parquet(out / "tracking.parquet"),
        },
    )
    write_json(out / "plays.json", metadata)
    write_json(
        out / "config.json",
        {
            **config,
            "actual_epochs": epochs,
            "synthetic_games": games,
            "synthetic_plays_per_game": plays,
        },
    )
    examples: dict[str, list[tuple[Features, np.ndarray, np.ndarray]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    samples = []
    for meta in metadata:
        play_rows = rows.filter(
            (pl.col("game_id") == meta["game_id"]) & (pl.col("play_id") == meta["play_id"])
        )
        for cutoff in config["cutoffs"]:
            for horizon in config["horizons"]:
                sample = input_from_rows(play_rows, meta, cutoff, horizon, dataset_id)
                f = build_features(sample)
                y, mask = targets(reveal_from_rows(play_rows, sample))
                partition = split["games"][meta["game_id"]]
                examples[partition].append((f, y, mask))
                samples.append((sample, f, y, mask, partition, meta))
    manifest = {
        "dataset_id": dataset_id,
        "split_id": split["split_id"],
        "common_cohort": True,
        "samples_by_split": {s: len(es) for s, es in examples.items()},
        "sample_ids": [
            {
                "game": s.game_id,
                "play": s.play_id,
                "cutoff": s.cutoff,
                "horizon": s.horizon,
                "input_hash": s.input_hash(),
                "observed_targets": int(mask.sum()),
                "split": partition,
            }
            for s, _, _, mask, partition, _ in samples
        ],
    }
    write_json(out / "cohort.json", manifest)
    bundles = {}
    for family in FAMILIES:
        trial_config = {
            "family": family,
            "seed": 1729,
            "epochs": epochs,
            "device": "cpu",
            "dataset_id": dataset_id,
            "split_id": split["split_id"],
            "config_hash": digest(config),
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_tree_hash": source_hash,
            "working_tree_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
            ),
            "scope": "synthetic software validation",
            "max_seconds": 120,
        }
        identity = registry.start(trial_config)
        start = time.perf_counter()
        try:
            if family in FAMILIES[:3]:
                model = Baseline(family)
                model.fit(examples["train"])
                history = []
                params = None
            else:
                width = matched_width() if family == "pooled" else 24
                model, history = train_neural(
                    family, examples["train"], examples["validation"], width, epochs, 120
                )
                params = parameter_count(model)
            means, covs, ys = [], [], []
            for f, y, mask in examples["validation"]:
                mean, cov = predict(model, f)
                means.append(mean[mask])
                covs.append(cov[mask])
                ys.append(y[mask])
            scale = fit_scale(np.concatenate(means), np.concatenate(covs), np.concatenate(ys))
            threshold = float(
                np.quantile(np.trace(np.concatenate(covs) * scale, axis1=1, axis2=2), 0.25)
            )
            bundle = {
                "model": model,
                "scale": scale,
                "sharp_threshold": threshold,
                "calibration_id": digest(
                    {
                        "scale": scale,
                        "validation_games": [
                            g for g, s in split["games"].items() if s == "validation"
                        ],
                    }
                ),
                "model_id": identity,
                "split_id": split["split_id"],
                "dataset_id": dataset_id,
            }
            bundles[family] = bundle
            joblib.dump(bundle, out / f"{family}.joblib")
            registry.finish(
                identity,
                "succeeded",
                {
                    "seconds": time.perf_counter() - start,
                    "parameters": params,
                    "history": history,
                    "calibration_scale": scale,
                    "checkpoint": str(out / f"{family}.joblib"),
                    "checkpoint_sha256": file_hash(out / f"{family}.joblib"),
                    "dtype": "float32",
                    "peak_cuda_bytes": None,
                },
            )
        except BaseException as error:
            registry.finish(
                identity,
                "cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                {
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "seconds": time.perf_counter() - start,
                },
            )
            raise
    # Protected test exposure happens only after model/calibration choices are frozen.
    with (out / "test-exposure.json").open("x") as file:
        json.dump(
            {
                "scope": "synthetic only",
                "frozen_models": [b["model_id"] for b in bundles.values()],
                "selection": "no final-test selection",
                "exposure": 1,
            },
            file,
        )
    evaluations = []
    forecasts = []
    for sample, f, y, mask, partition, meta in samples:
        if partition == "train":
            continue
        for family, bundle in bundles.items():
            mean, cov = predict(bundle["model"], f)
            cov = cov * bundle["scale"]
            draw = gaussian_samples(mean, cov, 64, 1729)
            score = energy_score(draw[:, mask], y[mask])
            nll = gaussian_nll(mean[mask], cov[mask], y[mask])
            record = {
                "game_id": sample.game_id,
                "play_id": sample.play_id,
                "cutoff": sample.cutoff,
                "horizon": sample.horizon,
                "family": family,
                "split": partition,
                "energy_score": float(score.mean()),
                "nll": float(nll.mean()),
                "endpoint_error": float(np.linalg.norm(mean[mask] - y[mask], axis=1).mean()),
                "ambiguity_trace_yards2": float(np.trace(cov, axis1=1, axis2=2).mean()),
                "marginal_coverage_80": coverage(mean[mask], cov[mask], y[mask], 0.8),
                "sharp_failed_defenders": int(
                    (
                        (np.trace(cov, axis1=1, axis2=2) <= bundle["sharp_threshold"])
                        & (mahalanobis(mean, cov, y) > 3.218875825)
                        & mask
                    ).sum()
                ),
                "targets": int(mask.sum()),
                "offense": meta["offense"],
                "defense": meta["defense"],
            }
            evaluations.append(record)
            forecasts.append(
                {
                    **{
                        k: record[k]
                        for k in ["game_id", "play_id", "cutoff", "horizon", "family", "split"]
                    },
                    "mean": mean.tolist(),
                    "covariance": cov.tolist(),
                }
            )
    pl.DataFrame(evaluations).write_parquet(out / "scores.parquet")
    write_json(out / "forecasts.json", forecasts)
    primary = [
        r
        for r in evaluations
        if r["split"] == "test" and r["cutoff"] == -0.1 and r["horizon"] == 1.0
    ]
    aggregate = []
    for family in FAMILIES:
        rs = [r for r in primary if r["family"] == family]
        aggregate.append(
            {
                "family": family,
                "plays": len(rs),
                "games": len({r["game_id"] for r in rs}),
                **{
                    k: float(np.mean([r[k] for r in rs]))
                    for k in ["energy_score", "nll", "endpoint_error", "marginal_coverage_80"]
                },
            }
        )
    graph = [r for r in primary if r["family"] == "graph"]
    pooled = [r for r in primary if r["family"] == "pooled"]
    paired = clustered_interval(
        [r["game_id"] for r in graph],
        [g["energy_score"] - p["energy_score"] for g, p in zip(graph, pooled, strict=True)],
    )
    timing = []
    for family in FAMILIES:
        for meta in metadata:
            if split["games"][meta["game_id"]] != "test":
                continue
            matching = [
                r
                for r in forecasts
                if r["game_id"] == meta["game_id"]
                and r["play_id"] == meta["play_id"]
                and r["family"] == family
                and r["horizon"] == 1.0
            ]
            early = next(r for r in matching if r["cutoff"] == -1.0)
            late = next(r for r in matching if r["cutoff"] == -0.1)
            scores = [
                r
                for r in evaluations
                if r["game_id"] == meta["game_id"]
                and r["play_id"] == meta["play_id"]
                and r["family"] == family
                and r["horizon"] == 1.0
            ]
            e = next(r for r in scores if r["cutoff"] == -1.0)
            late_score = next(r for r in scores if r["cutoff"] == -0.1)
            timing.append(
                {
                    "family": family,
                    "game_id": meta["game_id"],
                    "play_id": meta["play_id"],
                    "belief_change_nats": belief_change(
                        np.array(early["mean"]),
                        np.array(early["covariance"]),
                        np.array(late["mean"]),
                        np.array(late["covariance"]),
                    ),
                    "late_minus_early_energy": late_score["energy_score"] - e["energy_score"],
                    "log_density_improvement": e["nll"] - late_score["nll"],
                }
            )
    report = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "split_id": split["split_id"],
        "scope": "Synthetic software validation. No empirical NFL/Bengals findings.",
        "primary": aggregate,
        "graph_minus_pooled": paired,
        "timing": timing,
        "trials": registry.records(),
        "verified_nfl_games": 0,
        "verified_bengals_plays": 0,
        "cases": select_cases(primary),
        "budget": {"cloud_usd": 0, "max_trials": 24, "max_epochs": 40},
    }
    write_json(out / "report.json", report)
    write_json(root / "active.json", {"run_path": str(out.resolve())})
    return out


def select_cases(primary: list[dict]) -> list[dict]:
    candidates = [r for r in primary if r["family"] == "geometry" and r["offense"] == "CIN"]
    if not candidates:
        return []
    ordinary = candidates[np.random.default_rng(1729).integers(len(candidates))]
    spread_tied = len({r["ambiguity_trace_yards2"] for r in candidates}) == 1
    return [
        {
            "kind": kind,
            "game_id": record["game_id"],
            "play_id": record["play_id"],
            "synthetic": True,
            "split": "test",
            "note": (
                "Retrospective selection; all geometry spreads tie; "
                "first tied case, not uniquely ambiguous"
                if kind == "ambiguous" and spread_tied
                else "Retrospective selection; illustration only"
            ),
        }
        for kind, record in [
            ("ordinary", ordinary),
            ("ambiguous", max(candidates, key=lambda r: r["ambiguity_trace_yards2"])),
            ("surprising", max(candidates, key=lambda r: r["nll"])),
        ]
    ]


def forecast_sample(sample: ForecastInput, bundle: dict) -> Forecast:
    if sample.dataset_id != bundle["dataset_id"]:
        raise ValueError("model dataset mismatch")
    f = build_features(sample)
    model = bundle["model"]
    component_means = component_covs = weights = None
    if isinstance(model, SharedMixtureModel):
        model.eval()
        with torch.no_grad():
            m, v, w = model(f.graph())
        component_means = m.numpy()
        component_covs = v.numpy()
        weights = w.exp().numpy()[0]
        mean = (component_means * weights[None, :, None]).sum(1)
        # Covariance is a moment summary only; renderer uses component hypotheses.
        cov = np.zeros((len(mean), 2, 2))
    else:
        mean, cov = predict(model, f)
    c = [-1.0, -0.5, -0.1].index(sample.cutoff)
    h = [0.5, 1.0].index(sample.horizon)
    cov = cov * bundle.get("scales", {}).get((c, h), bundle["scale"])
    if weights is not None:
        assert component_covs is not None and component_means is not None
        component_covs = component_covs * bundle["scales"][(c, h)]
        delta = component_means - mean[:, None]
        cov = (
            weights[None, :, None, None]
            * (component_covs + delta[:, :, :, None] * delta[:, :, None, :])
        ).sum(1)
    if not np.isfinite(mean).all() or not np.isfinite(cov).all():
        raise FloatingPointError("non-finite forecast")
    defenders = []
    for i, node in enumerate(np.flatnonzero(f.defenders)):
        modes: tuple[DefenderMode, ...] = ()
        if weights is not None:
            assert component_means is not None and component_covs is not None
            modes = tuple(
                DefenderMode(
                    weight=float(weight),
                    mean_displacement=tuple(component_means[i, k]),
                    covariance=(tuple(component_covs[i, k, 0]), tuple(component_covs[i, k, 1])),
                )
                for k, weight in enumerate(weights)
            )
        defenders.append(
            DefenderForecast(
                entity_id=f.ids[node],
                anchor=tuple(f.anchor[node]),
                mean_displacement=tuple(mean[i]),
                covariance=(
                    (float(cov[i, 0, 0]), float(cov[i, 0, 1])),
                    (float(cov[i, 1, 0]), float(cov[i, 1, 1])),
                ),
                modes=modes,
            )
        )
    return Forecast(
        input_hash=sample.input_hash(),
        model_id=bundle["model_id"],
        dataset_id=sample.dataset_id,
        calibration_id=bundle["calibration_id"],
        feature_version=sample.feature_version,
        split_id=bundle["split_id"],
        cutoff=sample.cutoff,
        anchor_seconds=sample.anchor_seconds,
        horizon=sample.horizon,
        defenders=tuple(defenders),
        distribution="shared_component_mixture"
        if weights is not None
        else "marginal_bivariate_gaussian",
        region_note=(
            "Conditional component 80% ellipses; whole-mixture density-set coverage is in "
            "evaluation. Shared component weights are not scheme labels."
            if weights is not None
            else "Nominal 80% marginal region; inspect measured coverage. Not joint."
        ),
    )
