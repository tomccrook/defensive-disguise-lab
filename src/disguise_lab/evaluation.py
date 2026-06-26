"""Locked proper-score, calibration, timing and clustered empirical evaluation."""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from scipy.stats import chi2

from disguise_lab.audit import file_hash
from disguise_lab.empirical import CachedNFL, distributions, draws, marginal_nll, moments
from disguise_lab.probability import belief_change, clustered_interval, energy_score, mahalanobis
from disguise_lab.registry import Registry
from disguise_lab.study import FAMILIES, write_json


def scaled_prediction(data, bundle, combos, **kwargs):
    mean, cov, weights = distributions(data, bundle["model"], combos, **kwargs)
    multiplier = np.array([bundle["scales"][(int(c), int(h))] for _, c, h in combos])
    cov = cov * multiplier.reshape(-1, *([1] * (cov.ndim - 1)))
    return mean, cov, weights


def score_predictions(data, combos, mean, cov, weights, bundle, config, protocol, partition):
    if not np.isfinite(mean).all() or not np.isfinite(cov).all():
        raise FloatingPointError("nonfinite predictive distribution")
    if weights is not None and (
        not np.isfinite(weights).all()
        or np.any(weights <= 0)
        or not np.allclose(weights.sum(-1), 1, atol=1e-5)
    ):
        raise ValueError("positive normalised mixture probabilities required")
    p, c, h = combos.T
    y = np.array(data.y[p, h])
    mask = np.array(data.mask[p, h])
    point, variance = moments(mean, cov, weights)
    nll = marginal_nll(mean, cov, y, weights)
    distance = np.linalg.norm(point - y, axis=-1)
    mahal = mahalanobis(point, variance, y)
    trace = np.trace(variance, axis1=-2, axis2=-1)
    energy = np.empty_like(nll)
    joint = np.full(len(p), np.nan)
    for offset in range(0, len(p), 32):
        stop = offset + 32
        n = len(p[offset:stop])
        samples = draws(
            mean[offset:stop],
            cov[offset:stop],
            config["budget"]["energy_samples"],
            config["seed"] + offset,
            None if weights is None else weights[offset:stop],
        )
        energy[offset:stop] = energy_score(samples, y[offset:stop].reshape(-1, 2)).reshape(n, 11)
        # Joint score is a secondary independent-Gaussian/shared-component model check.
        complete = mask[offset:stop].all(axis=1)
        if complete.any():
            joint_samples = samples.reshape(len(samples), n, 22)[:, complete]
            joint[offset:stop][complete] = energy_score(
                joint_samples, y[offset:stop][complete].reshape(-1, 22)
            )
    nominal_coverage = {q: (mahal <= chi2.ppf(q, 2)) for q in [0.5, 0.8, 0.9, 0.95]}
    if weights is not None:
        # Mixture regions use sampled marginal high-density sets, not moment ellipses.
        for offset in range(0, len(p), 16):
            stop = offset + 16
            n = len(p[offset:stop])
            w = weights[offset:stop]
            samples = draws(mean[offset:stop], cov[offset:stop], 256, config["seed"] + offset, w)
            sample_y = samples.reshape(256, n, 11, 2)
            from scipy.special import logsumexp

            from disguise_lab.probability import gaussian_nll

            log_density = logsumexp(
                np.log(w)[None, :, None, :]
                - gaussian_nll(
                    mean[offset:stop][None], cov[offset:stop][None], sample_y[:, :, :, None, :]
                ),
                axis=-1,
            )
            for q in nominal_coverage:
                threshold = np.quantile(log_density, 1 - q, axis=0)
                nominal_coverage[q][offset:stop] = -nll[offset:stop] >= threshold
    records = []
    for i, (play, a, b) in enumerate(combos):
        valid = mask[i]
        n = int(valid.sum())
        meta = data.metadata[play]
        if not n:
            continue
        sharp = trace[i] <= bundle["sharp_thresholds"][(int(a), int(b))]
        records.append(
            {
                "game_id": meta["game_id"],
                "play_id": meta["play_id"],
                "cutoff": [-1.0, -0.5, -0.1][a],
                "horizon": [0.5, 1.0][b],
                "family": bundle["variant"],
                "split": partition,
                "protocol": protocol,
                "energy_score": float(energy[i, valid].mean()),
                "nll": float(nll[i, valid].mean()),
                "endpoint_error": float(distance[i, valid].mean()),
                "bias_x": float((point[i] - y[i])[valid, 0].mean()),
                "bias_y": float((point[i] - y[i])[valid, 1].mean()),
                "rmse": float(np.sqrt(((point[i] - y[i])[valid] ** 2).sum(-1).mean())),
                "ambiguity_trace_yards2": float(trace[i].mean()),
                "joint_energy_score": None if np.isnan(joint[i]) else float(joint[i]),
                "all_marginal_80_regions_cover": (
                    bool(nominal_coverage[0.8][i].all()) if valid.all() else None
                ),
                **{
                    f"marginal_coverage_{int(q * 100)}": float(value[i, valid].mean())
                    for q, value in nominal_coverage.items()
                },
                "sharp_failed_defenders": int((sharp & ~nominal_coverage[0.8][i] & valid).sum()),
                "targets": n,
                "offense": meta["offense"],
                "defense": meta["defense"],
                "region_method": "Gaussian marginal ellipse"
                if weights is None
                else "marginal mixture density set; 256-sample approximate threshold",
                "model_id": bundle["model_id"],
                "input_index": int(play),
            }
        )
    return records, point, variance


def aggregate(records):
    keys = [
        "energy_score",
        "nll",
        "endpoint_error",
        "marginal_coverage_50",
        "marginal_coverage_80",
        "marginal_coverage_90",
        "marginal_coverage_95",
        "ambiguity_trace_yards2",
        "rmse",
        "bias_x",
        "bias_y",
    ]
    return [
        {
            "family": family,
            "plays": len(selected),
            "games": len({r["game_id"] for r in selected}),
            **{k: float(np.mean([r[k] for r in selected])) for k in keys},
            "complete_joint_plays": sum(r["joint_energy_score"] is not None for r in selected),
            "joint_energy_score": float(
                np.mean(
                    [
                        r["joint_energy_score"]
                        for r in selected
                        if r["joint_energy_score"] is not None
                    ]
                )
            ),
            "all_marginal_80_regions_cover": float(
                np.mean(
                    [
                        r["all_marginal_80_regions_cover"]
                        for r in selected
                        if r["all_marginal_80_regions_cover"] is not None
                    ]
                )
            ),
        }
        for family in dict.fromkeys(r["family"] for r in records)
        if (selected := [r for r in records if r["family"] == family])
    ]


def paired(records, first, second, config):
    a = {(r["game_id"], r["play_id"]): r for r in records if r["family"] == first}
    b = {(r["game_id"], r["play_id"]): r for r in records if r["family"] == second}
    if a.keys() != b.keys():
        raise ValueError("paired models have different scoring cohorts")
    return clustered_interval(
        [k[0] for k in a],
        [a[k]["energy_score"] - b[k]["energy_score"] for k in a],
        draws=config["budget"]["bootstrap_games"],
    )


def cases(primary, selected_family, config):
    candidates = [r for r in primary if r["family"] == selected_family]
    ordinary = candidates[np.random.default_rng(config["seed"]).integers(len(candidates))]
    picks = [
        ("ordinary", ordinary),
        ("ambiguous", max(candidates, key=lambda r: r["ambiguity_trace_yards2"])),
        ("surprising", max(candidates, key=lambda r: r["nll"])),
    ]
    failed = [r for r in candidates if r["sharp_failed_defenders"] > 0]
    if failed:
        picks.append(("overconfident failure", max(failed, key=lambda r: r["nll"])))
    return [
        {
            "kind": kind,
            "game_id": r["game_id"],
            "play_id": r["play_id"],
            "synthetic": False,
            "split": "test",
            "model_family": selected_family,
            "offense": r["offense"],
            "defense": r["defense"],
            "note": (
                "Retrospective held-out 2022 case; illustrates a model forecast, "
                "not defensive intent."
                + (
                    " Predictive spread ties across all candidates; first tied case "
                    "does not establish an ambiguity ranking."
                    if kind == "ambiguous"
                    and len({x["ambiguity_trace_yards2"] for x in candidates}) == 1
                    else ""
                )
            ),
            "evidence": {
                k: r[k]
                for k in [
                    "energy_score",
                    "nll",
                    "endpoint_error",
                    "ambiguity_trace_yards2",
                    "sharp_failed_defenders",
                ]
            },
        }
        for kind, r in picks
    ]


def validation_evidence(data, bundle, split, config, protocol):
    indices = np.array(
        [i for i, m in enumerate(data.metadata) if split["games"][m["game_id"]] == "validation"]
    )
    combos = data.combinations(indices)
    calibration = []
    mean, cov, w = scaled_prediction(data, bundle, combos)
    p, c, h = combos.T
    scales = np.array([bundle["scales"][(int(a), int(b))] for _, a, b in combos])
    raw = cov / scales.reshape(-1, *([1] * (cov.ndim - 1)))
    for a in range(3):
        for b in range(2):
            take = (c == a) & (h == b)
            observed = data.mask[p[take], h[take]]
            target = data.y[p[take], h[take]]
            weight = None if w is None else w[take]
            before = marginal_nll(mean[take], raw[take], target, weight)
            after = marginal_nll(mean[take], cov[take], target, weight)
            diagnostic = {
                "protocol": protocol,
                "family": bundle["variant"],
                "cutoff": [-1, -0.5, -0.1][a],
                "horizon": [0.5, 1][b],
                "targets": int(observed.sum()),
                "scale": float(scales[take][0]),
                "nll_before": float(before[observed].mean()),
                "nll_after": float(after[observed].mean()),
            }
            if w is None:
                for label, variance in [("before", raw[take]), ("after", cov[take])]:
                    distance = mahalanobis(mean[take], variance, target)
                    diagnostic[f"coverage_80_{label}"] = float(
                        (distance[observed] <= chi2.ppf(0.8, 2)).mean()
                    )
            calibration.append(diagnostic)
    # Selection is on the primary condition only; no redundant six-way MC sweep.
    primary_condition = (c == 2) & (h == 1)
    values, _, _ = score_predictions(
        data,
        combos[primary_condition],
        mean[primary_condition],
        cov[primary_condition],
        None if w is None else w[primary_condition],
        bundle,
        config,
        protocol,
        "validation",
    )
    return values, calibration


def evaluate(data_root: Path, out: Path, config: dict) -> dict:
    if (out / "report.json").exists():
        return json.loads((out / "report.json").read_text())
    if not (out / "training-complete.json").exists():
        raise ValueError("both protocols must finish training before final evaluation")
    data = CachedNFL(data_root)
    from disguise_lab.robustness import PLAN, run_checks

    analysis_plan = out / "analysis-plan.json"
    if not analysis_plan.exists() or json.loads(analysis_plan.read_text()) != PLAN:
        raise ValueError("secondary analysis must be registered before test exposure")
    prereg = json.loads((out / "preregistration.json").read_text())
    if prereg["config"] != config:
        raise ValueError("evaluation config differs from frozen protocol")
    if file_hash(data_root / "dataset.json") != prereg["dataset_hash"]:
        raise ValueError("canonical manifest changed after registration")
    for name, expected in data.dataset["feature_cache_hashes"].items():
        if file_hash(data_root / f"{name}.npy") != expected:
            raise ValueError(f"canonical array changed after registration: {name}")
    bundles = {
        protocol: {p.stem: joblib.load(p) for p in (out / protocol).glob("*.joblib")}
        for protocol in prereg["splits"]
    }
    expected_variants = {
        "chronological": set(FAMILIES)
        | {
            "geometry_static",
            "temporal_static",
            "graph_no_side",
            "pooled_mixture",
            "graph_mixture",
            "pooled_seed1730",
            "graph_seed1730",
        },
        "opponent": set(FAMILIES),
    }
    registry = Registry(Path("artifacts/trials.sqlite"), config["budget"]["total_registry_trials"])
    successful = {r["trial_id"]: r for r in registry.records() if r["status"] == "succeeded"}
    for protocol, models in bundles.items():
        if set(models) != expected_variants[protocol]:
            raise ValueError("registered model set incomplete or unexpected")
        for variant, bundle in models.items():
            trial = successful.get(bundle["model_id"])
            if not trial or (
                trial["result"]["checkpoint_sha256"]
                != file_hash(out / protocol / f"{variant}.joblib")
                or bundle["dataset_id"] != data.dataset["dataset_id"]
                or bundle["split_id"] != prereg["splits"][protocol]["split_id"]
            ):
                raise ValueError("model checkpoint does not match frozen successful trial")
    records = []
    predictions = {}
    calibration = []
    validation_primary: list[dict] = []
    for protocol, split in prereg["splits"].items():
        for variant, bundle in bundles[protocol].items():
            cache = out / protocol / f"{variant}-validation.json"
            fingerprint = {
                "checkpoint_hash": file_hash(out / protocol / f"{variant}.joblib"),
                "dataset_hash": prereg["dataset_hash"],
                "evaluation_source_hash": file_hash(Path(__file__)),
            }
            cached = json.loads(cache.read_text()) if cache.exists() else None
            if cached and cached["fingerprint"] == fingerprint:
                values, diagnostics = cached["records"], cached["calibration"]
            else:
                values, diagnostics = validation_evidence(data, bundle, split, config, protocol)
                write_json(
                    cache,
                    {"fingerprint": fingerprint, "records": values, "calibration": diagnostics},
                )
            calibration.extend(diagnostics)
            records.extend(values)
            if protocol == "chronological":
                validation_primary.extend(values)
    validation_aggregate = aggregate(validation_primary)
    serving = list(FAMILIES) + ["pooled_mixture", "graph_mixture"]
    best = min(
        [r for r in validation_aggregate if r["family"] in serving],
        key=lambda r: r["energy_score"],
    )["family"]
    frozen = {
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "evaluation_source_sha256": file_hash(Path(__file__)),
        "hypothesis_registry_sha256": file_hash(Path("configs/hypotheses.json")),
        "secondary_plan_sha256": file_hash(analysis_plan),
        "selected_demo_family": best,
        "selection": (
            "validation energy; six Gaussian families and two preregistered shared-mixture heads"
        ),
        "all_models": {
            protocol: {
                v: {
                    "model_id": b["model_id"],
                    "checkpoint_sha256": file_hash(out / protocol / f"{v}.joblib"),
                    "calibration_id": b["calibration_id"],
                }
                for v, b in bs.items()
            }
            for protocol, bs in bundles.items()
        },
        "preregistration_hash": file_hash(out / "preregistration.json"),
    }
    write_json(out / "model-freeze.json", frozen)
    exposure = out / "test-exposure.json"
    if exposure.exists():
        raise ValueError(
            "test exposure already recorded; preserve failed evaluation "
            "before explicit corrective rerun"
        )
    with exposure.open("x") as stream:
        json.dump(
            {
                "exposure": 1,
                "started_at_utc": datetime.now(UTC).isoformat(),
                "model_freeze_hash": file_hash(out / "model-freeze.json"),
                "reason": "one locked pass after fitting/calibration/selection; no test tuning",
            },
            stream,
            indent=2,
        )
    for protocol, split in prereg["splits"].items():
        indices = np.array(
            [i for i, m in enumerate(data.metadata) if split["games"][m["game_id"]] == "test"]
        )
        combos = data.combinations(indices)
        for variant, bundle in bundles[protocol].items():
            mean, cov, w = scaled_prediction(data, bundle, combos)
            values, point, variance = score_predictions(
                data, combos, mean, cov, w, bundle, config, protocol, "test"
            )
            records.extend(values)
            pl.DataFrame(values).write_parquet(out / protocol / f"{variant}-test-scores.parquet")
            np.savez_compressed(
                out / protocol / f"{variant}-test-distributions.npz",
                combos=combos,
                mean=mean,
                covariance=cov,
                weights=np.array([]) if w is None else w,
            )
            if protocol == "chronological" and variant in FAMILIES:
                predictions[variant] = (combos, point, variance)
            print(f"locked scores: {protocol}/{variant}", flush=True)
    pl.DataFrame(records).write_parquet(out / "scores.parquet")
    primary = [
        r
        for r in records
        if r["protocol"] == "chronological"
        and r["split"] == "test"
        and r["cutoff"] == -0.1
        and r["horizon"] == 1.0
    ]
    opponent = [
        r
        for r in records
        if r["protocol"] == "opponent"
        and r["split"] == "test"
        and r["cutoff"] == -0.1
        and r["horizon"] == 1.0
    ]
    main = [r for r in primary if r["family"] in FAMILIES]
    lookup = {
        (r["family"], r["input_index"], r["cutoff"], r["horizon"]): r
        for r in records
        if r["protocol"] == "chronological" and r["split"] == "test"
    }
    timing = []
    for family, (combos, point, variance) in predictions.items():
        for i in range(0, len(combos), 6):
            p = int(combos[i, 0])
            early = i + 1
            late = i + 5
            meta = data.metadata[p]
            early_record = lookup.get((family, p, -1.0, 1.0))
            late_record = lookup.get((family, p, -0.1, 1.0))
            if early_record is None or late_record is None:
                continue
            timing.append(
                {
                    "family": family,
                    "game_id": meta["game_id"],
                    "play_id": meta["play_id"],
                    "belief_change_nats": belief_change(
                        point[early], variance[early], point[late], variance[late]
                    ),
                    "late_minus_early_energy": late_record["energy_score"]
                    - early_record["energy_score"],
                    "log_density_improvement": early_record["nll"] - late_record["nll"],
                }
            )
    registry = Registry(Path("artifacts/trials.sqlite"), config["budget"]["total_registry_trials"])
    test_indices = np.array(
        [
            i
            for i, m in enumerate(data.metadata)
            if prereg["splits"]["chronological"]["games"][m["game_id"]] == "test"
        ]
    )
    checks = run_checks(data, bundles["chronological"], test_indices, config, timing, primary)
    from disguise_lab.strata import formation_strata

    formations = formation_strata(
        data.metadata,
        prereg["splits"]["chronological"],
        primary,
        Path("data/raw/nfl-big-data-bowl-2025"),
    )
    report = {
        "run_id": out.name,
        "dataset_id": data.dataset["dataset_id"],
        "split_id": prereg["splits"]["chronological"]["split_id"],
        "scope": (
            "Historical NFL 2022 weeks 1–9; private local offline snap-conditioned study. "
            "Idealised tracking, no intent/quarterback-perception claim."
        ),
        "primary": aggregate(main),
        "all_variants": aggregate(primary),
        "validation": validation_aggregate,
        "validation_calibration": calibration,
        "graph_minus_pooled": paired(main, "graph", "pooled", config),
        "opponent_primary": aggregate(opponent),
        "opponent_graph_minus_pooled": paired(opponent, "graph", "pooled", config),
        "ablations": {
            variant: paired(primary, variant, base, config)
            for variant, base in [
                ("geometry_static", "geometry"),
                ("temporal_static", "temporal"),
                ("graph_no_side", "graph"),
                ("graph_mixture", "pooled_mixture"),
                ("graph_seed1730", "pooled_seed1730"),
            ]
        },
        "timing": timing,
        "robustness": checks,
        "formation_audit": formations,
        "timing_intervals": {
            family: clustered_interval(
                [r["game_id"] for r in timing if r["family"] == family],
                [r["late_minus_early_energy"] for r in timing if r["family"] == family],
                draws=config["budget"]["bootstrap_games"],
            )
            for family in FAMILIES
        },
        "cases": cases(primary, best, config),
        "selected_demo_family": best,
        "verified_nfl_games": data.dataset["games"],
        "verified_bengals_plays": data.dataset["bengals_offense_plays"]
        + data.dataset["bengals_defense_plays"],
        "budget": config["budget"],
        "trials": registry.records(),
        "test_exposure": 1,
        "model_freeze": frozen,
        "team_slices": {
            view: aggregate([r for r in main if r[field] == "CIN"])
            for view, field in [("bengals_offense", "offense"), ("bengals_defense", "defense")]
        },
    }
    # Serving uses a single frozen primary protocol; detailed controls stay in the report.
    for family in serving:
        joblib.dump(bundles["chronological"][family], out / f"{family}.joblib")
    report["serving_hashes"] = {family: file_hash(out / f"{family}.joblib") for family in serving}
    write_json(out / "dataset.json", {**data.dataset, "canonical_root": str(data_root.resolve())})
    write_json(out / "plays.json", data.metadata)
    write_json(out / "split.json", prereg["splits"]["chronological"])
    write_json(
        out / "cohort.json",
        {
            "dataset_id": data.dataset["dataset_id"],
            "split_id": report["split_id"],
            "common_cohort": True,
            "plays": len(data.metadata),
            "targets_by_horizon": data.dataset["observed_defender_targets_by_horizon"],
            "exclusions_hash": file_hash(data_root / "exclusions.json"),
        },
    )
    write_json(out / "report.json", report)
    write_json(Path("artifacts/active.json"), {"run_path": str(out.resolve())})
    return report
