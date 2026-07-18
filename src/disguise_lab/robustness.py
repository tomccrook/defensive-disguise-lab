"""Predeclared post-freeze sensitivity checks; no model or threshold selection."""

from dataclasses import replace
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from torch_geometric.data import Batch

from disguise_lab.empirical import draws
from disguise_lab.features import WIDTH
from disguise_lab.models import Baseline
from disguise_lab.probability import clustered_interval, energy_score

PLAN: dict[str, Any] = {
    "registered_before_test": True,
    "purpose": "secondary descriptive model stress; no test-dependent fitting",
    "subset": "seed 1729 choose min(256, available) chronological test plays; primary -0.1/+1",
    "models": ["geometry", "pooled", "graph"],
    "sensor_noise": (
        "Independent constant per-player coordinate offsets SD 0.1 yd; constrain all "
        "observed coordinates inside field; translate anchors too; recompute spatial "
        "edges and pooled geometry; unchanged velocities because offsets constant"
    ),
    "short_history": (
        "GRU sees final six causal frames; observed -1s anchor retained explicitly. "
        "Model stress, not accepted production input."
    ),
    "edges": ["complete directed", "distance <=10 yd", "distance <=20 yd"],
    "mc_precision": (
        "First 128 randomly selected plays; 64/128 draws and four fixed seeds; "
        "report graph-minus-pooled variability"
    ),
    "formation": (
        "Source offence formation is only an audit stratum; never input; "
        "require >=50 plays and >=3 games"
    ),
    "concepts": (
        "Spread vs symmetric KL vs realised NLL: Spearman and top-decile overlap, descriptive only"
    ),
    "tuning": False,
}


def shifted_features(f, offset):
    # Bound a time-constant sensor bias, using observed coordinates only.
    relative = f.history[:, :, :2] * 10
    low = np.minimum(relative.min(1), f.anchor - f.last + relative[:, -1])
    high = np.maximum(relative.max(1), f.anchor - f.last + relative[:, -1])
    origin = f.last - relative[:, -1]
    offset = np.clip(offset, -origin - low, np.array([120, WIDTH]) - origin - high)
    history = f.history.copy()
    history[:, :, :2] += offset[:, None] / 10
    last = f.last + offset
    send, receive = f.edge_index
    delta = last[send] - last[receive]
    attr = f.edge_attr.copy()
    attr[:, :2] = delta / 10
    attr[:, 2] = np.linalg.norm(delta, axis=-1) / 10
    return replace(f, history=history, anchor=f.anchor + offset, last=last, edge_attr=attr)


def stress_prediction(data, bundle, combos, mode, seed):
    rng = np.random.default_rng(seed)
    mean: list[np.ndarray] = []
    cov: list[np.ndarray] = []
    model = bundle["model"]
    for start in range(0, len(combos), 64):
        fs, corrections = [], []
        for p, c, h in combos[start : start + 64]:
            original = data.features(int(p), int(c), int(h))
            f = original
            if mode == "sensor_bias_0.1yd":
                f = shifted_features(f, rng.normal(0, 0.1, (22, 2)))
            elif mode == "six_frame_history":
                f = replace(f, history=f.history[:, -6:])
            elif mode.startswith("radius_"):
                keep = f.edge_attr[:, 2] <= float(mode.split("_")[1]) / 10
                f = replace(f, edge_index=f.edge_index[:, keep], edge_attr=f.edge_attr[keep])
            fs.append(f)
            corrections.append((f.anchor - original.anchor)[f.defenders])
        if isinstance(model, Baseline):
            values = [model.predict(f) for f in fs]
            m = np.stack([v[0] for v in values])
            v = np.stack([v[1] for v in values])
        else:
            model.eval()
            with torch.no_grad():
                m0, v0 = model(Batch.from_data_list([f.graph() for f in fs]))
            m = m0.numpy().reshape(-1, 11, 2)
            v = v0.numpy().reshape(-1, 11, 2, 2)
        mean.extend(m + np.array(corrections))
        cov.extend(v)
    multiplier = np.array([bundle["scales"][(int(c), int(h))] for _, c, h in combos])
    return np.array(mean), np.array(cov) * multiplier[:, None, None, None], None


def subset_energy(data, combos, prediction, count, seed):
    p, _, h = combos.T
    mean, cov, w = prediction
    values: list[float] = []
    for start in range(0, len(p), 16):
        end = start + 16
        samples = draws(
            mean[start:end],
            cov[start:end],
            count,
            seed + start,
            None if w is None else w[start:end],
        )
        score = energy_score(samples, data.y[p[start:end], h[start:end]].reshape(-1, 2))
        score = score.reshape(-1, 11)
        mask = data.mask[p[start:end], h[start:end]]
        values.extend(np.array([s[k].mean() for s, k in zip(score, mask, strict=True)]))
    return np.array(values)


def run_checks(data, bundles, test_indices, config, timing, primary):
    selected = np.random.default_rng(config["seed"]).permutation(test_indices)[:256]
    selected = selected[data.mask[selected, 1].any(1)]
    combos = np.column_stack([selected, np.full(len(selected), 2), np.ones(len(selected))]).astype(
        int
    )
    games = [data.metadata[p]["game_id"] for p in selected]
    checks = []
    predictions = {}
    for name in PLAN["models"]:
        for mode in ["unmodified", "sensor_bias_0.1yd", "six_frame_history"] + (
            ["radius_10", "radius_20"] if name == "graph" else []
        ):
            pred = stress_prediction(data, bundles[name], combos, mode, config["seed"])
            scores = subset_energy(data, combos, pred, 64, config["seed"])
            if mode == "unmodified":
                predictions[name] = pred
                reference = scores
            checks.append(
                {
                    "family": name,
                    "stress": mode,
                    "plays": len(selected),
                    "energy_score": float(scores.mean()),
                    "change_from_unmodified": clustered_interval(
                        games, (scores - reference).tolist(), draws=2000
                    ),
                }
            )
    precision = []
    short = combos[:128]
    for count in [64, 128]:
        for seed in [1729, 1730, 1731, 1732]:
            scores = {}
            for name in ["graph", "pooled"]:
                pred = tuple(None if x is None else x[:128] for x in predictions[name])
                scores[name] = subset_energy(data, short, pred, count, seed)
            precision.append(
                {
                    "draws": count,
                    "seed": seed,
                    "graph_minus_pooled": float((scores["graph"] - scores["pooled"]).mean()),
                }
            )
    concepts = {}
    for name in ["geometry", "pooled", "graph"]:
        rows = {(r["game_id"], r["play_id"]): r for r in primary if r["family"] == name}
        joined = [(r, rows[(r["game_id"], r["play_id"])]) for r in timing if r["family"] == name]
        arrays = {
            "belief_change": np.array([a["belief_change_nats"] for a, b in joined]),
            "ambiguity": np.array([b["ambiguity_trace_yards2"] for a, b in joined]),
            "surprise": np.array([b["nll"] for a, b in joined]),
        }
        pairs = {}
        for a, b in [
            ("ambiguity", "belief_change"),
            ("ambiguity", "surprise"),
            ("belief_change", "surprise"),
        ]:
            k = max(1, int(np.ceil(len(joined) * 0.1)))
            first, second = set(np.argsort(arrays[a])[-k:]), set(np.argsort(arrays[b])[-k:])
            rho = spearmanr(arrays[a], arrays[b]).statistic
            tied = np.ptp(arrays[a]) < 1e-12 or np.ptp(arrays[b]) < 1e-12
            pairs[f"{a}_vs_{b}"] = {
                "spearman": None if not np.isfinite(rho) else float(rho),
                "top_decile_overlap_fraction": None if tied else len(first & second) / k,
                "tie_note": "Undefined for constant scores" if tied else "Rank-defined top deciles",
            }
        concepts[name] = pairs
    return {
        "plan": PLAN,
        "stress": checks,
        "monte_carlo_precision": precision,
        "concept_distinctions": concepts,
        "scope": "secondary; fixed models; no retuning",
    }
