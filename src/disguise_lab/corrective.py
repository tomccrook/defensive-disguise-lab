"""Preserved post-exposure anchor adjustment for stronger geometry controls."""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from disguise_lab.audit import file_hash
from disguise_lab.contracts import digest
from disguise_lab.empirical import CachedNFL, calibrate, distributions
from disguise_lab.evaluation import aggregate, paired, scaled_prediction, score_predictions
from disguise_lab.models import Baseline
from disguise_lab.registry import Registry
from disguise_lab.study import write_json


def run(data_root: Path, out: Path):
    plan = json.loads(Path("configs/baseline-correction.json").read_text())
    prereg = json.loads((out / "preregistration.json").read_text())
    config, split = prereg["config"], prereg["splits"]["chronological"]
    report = json.loads((out / "report.json").read_text())
    destination = out / "baseline-correction"
    destination.mkdir(exist_ok=True)
    if (destination / "report.json").exists():
        return json.loads((destination / "report.json").read_text())
    if any(destination.iterdir()):
        raise RuntimeError(
            "Incomplete corrective audit preserved; do not overwrite its registration, "
            "checkpoints or exposure. Use an explicitly documented new study run."
        )
    if report["test_exposure"] != 1:
        raise ValueError("Correction requires the completed original locked test pass")
    data = CachedNFL(data_root)
    if data.dataset["dataset_id"] != report["dataset_id"]:
        raise ValueError("Correction must retain the original dataset")
    write_json(
        destination / "registration.json",
        {
            "plan": plan,
            "original_model_freeze_hash": file_hash(out / "model-freeze.json"),
            "original_test_exposure_hash": file_hash(out / "test-exposure.json"),
            "original_report_hash": file_hash(out / "report.json"),
        },
    )
    groups = {
        s: data.combinations(
            np.array([i for i, m in enumerate(data.metadata) if split["games"][m["game_id"]] == s])
        )
        for s in ["train", "validation", "test"]
    }
    registry = Registry(Path("artifacts/trials.sqlite"), 36)
    all_scores = []
    for variant in plan["variants"]:
        checkpoint = destination / f"{variant}.joblib"
        trial = registry.start(
            {
                "variant": variant,
                "protocol": "post_exposure_anchor_correction",
                "dataset_id": data.dataset["dataset_id"],
                "split_id": split["split_id"],
                "config_hash": digest(plan),
                "seed": 1729,
                "test_exposed_before_registration": True,
            }
        )
        started = time.perf_counter()
        try:
            model = Baseline(
                "geometry", static_geometry=variant == "geometry_static_residual", iterations=250
            )
            model.anchor_residual = True
            p, c, h = groups["train"].T
            target = data.y[p, h] - (data.last[p, c, 11:] - data.anchor[p, 11:])
            mask = data.mask[p, h]
            x = data.matrix(groups["train"], model.static_geometry)
            model.estimator.fit(x[mask], target[mask])
            model.context_covariance = {(a, b): np.eye(2) for a in range(3) for b in range(2)}
            mean, _, _ = distributions(data, model, groups["train"])
            for a in range(3):
                for b in range(2):
                    select = (c == a) & (h == b)
                    residual = (data.y[p[select], h[select]] - mean[select])[mask[select]]
                    model.context_covariance[(a, b)] = (
                        residual.T @ residual / len(residual) + np.eye(2) * 0.15**2
                    )
            model.covariance = model.context_covariance[(2, 1)]
            pred = distributions(data, model, groups["validation"])
            scales, sharp = calibrate(data, groups["validation"], *pred)
            bundle = {
                "model": model,
                "model_id": trial,
                "variant": variant,
                "protocol": "chronological",
                "dataset_id": data.dataset["dataset_id"],
                "split_id": split["split_id"],
                "scales": scales,
                "scale": scales[(2, 1)],
                "sharp_thresholds": sharp,
                "calibration_id": digest({str(k): v for k, v in scales.items()}),
                "post_exposure_correction": True,
            }
            joblib.dump(bundle, checkpoint)
            for partition in ["validation", "test"]:
                combos = groups[partition]
                if partition == "validation":
                    combos = combos[(combos[:, 1] == 2) & (combos[:, 2] == 1)]
                prediction = scaled_prediction(data, bundle, combos)
                values, _, _ = score_predictions(
                    data,
                    combos,
                    prediction[0],
                    prediction[1],
                    prediction[2],
                    bundle,
                    config,
                    "post_exposure_correction",
                    partition,
                )
                pl.DataFrame(values).write_parquet(
                    destination / f"{variant}-{partition}-scores.parquet"
                )
                all_scores.extend(values)
                if partition == "test":
                    np.savez_compressed(
                        destination / f"{variant}-test-distributions.npz",
                        combos=combos,
                        mean=prediction[0],
                        covariance=prediction[1],
                    )
            registry.finish(
                trial,
                "succeeded",
                {
                    "seconds": time.perf_counter() - started,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": file_hash(checkpoint),
                    "status": "post-exposure corrective; not untouched test evidence",
                },
            )
            print("Corrective audit completed:", variant, flush=True)
        except BaseException as error:
            registry.finish(
                trial, "failed", {"error_type": type(error).__name__, "message": str(error)}
            )
            raise
    original = (
        pl.read_parquet(out / "scores.parquet")
        .filter(
            (pl.col("protocol") == "chronological")
            & (pl.col("split") == "test")
            & (pl.col("cutoff") == -0.1)
            & (pl.col("horizon") == 1)
        )
        .to_dicts()
    )
    primary = [
        r for r in all_scores if r["split"] == "test" and r["cutoff"] == -0.1 and r["horizon"] == 1
    ]
    value = {
        "registration": plan,
        "primary": aggregate(primary),
        "validation": aggregate([r for r in all_scores if r["split"] == "validation"]),
        "comparisons": {
            "graph_minus_geometry_residual": paired(
                original + primary, "graph", "geometry_residual", config
            ),
            "history_minus_static_residual": paired(
                primary, "geometry_residual", "geometry_static_residual", config
            ),
        },
        "original_primary_unchanged": report["graph_minus_pooled"],
        "test_policy": "Post-exposure correction; original test pass and model selection preserved",
    }
    write_json(destination / "report.json", value)
    write_json(Path("docs/evidence/nfl-baseline-correction.json"), value)
    return value
