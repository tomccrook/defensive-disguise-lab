"""Reproducible static research figures and small evidence exports (synthetic only)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from disguise_lab.probability import coverage, energy_score, gaussian_samples
from disguise_lab.study import write_json


def export(run: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    report = json.loads((run / "report.json").read_text())
    if not json.loads((run / "dataset.json").read_text())["synthetic"]:
        raise ValueError("public report exporter accepts synthetic evidence only")
    for name in ["dataset", "split", "cohort", "report"]:
        write_json(
            destination / f"synthetic-{name}.json", json.loads((run / f"{name}.json").read_text())
        )
    write_json(destination / "trial-history.json", report["trials"])
    scores = pl.read_parquet(run / "scores.parquet")
    scores.filter(
        (pl.col("split") == "test") & (pl.col("cutoff") == -0.1) & (pl.col("horizon") == 1.0)
    ).write_csv(destination / "synthetic-primary-play-scores.csv")
    forecasts = json.loads((run / "forecasts.json").read_text())
    rows = pl.read_parquet(run / "tracking.parquet")
    calibration, precision = [], []
    for forecast in forecasts:
        if forecast["cutoff"] != -0.1 or forecast["horizon"] != 1.0:
            continue
        play_rows = rows.filter(
            (pl.col("game_id") == forecast["game_id"])
            & (pl.col("play_id") == forecast["play_id"])
            & (pl.col("side") == "defense")
        )
        anchor = play_rows.filter((pl.col("t") + 1).abs() < 1e-6).sort("entity_id")
        target = play_rows.filter((pl.col("t") - 1).abs() < 1e-6).sort("entity_id")
        if anchor["entity_id"].to_list() != target["entity_id"].to_list():
            raise ValueError("evidence exporter requires complete aligned fixture targets")
        y = target.select("x", "y").to_numpy() - anchor.select("x", "y").to_numpy()
        mean, cov = np.array(forecast["mean"]), np.array(forecast["covariance"])
        for level in [0.5, 0.8, 0.95]:
            calibration.append(
                {
                    "family": forecast["family"],
                    "split": forecast["split"],
                    "game_id": forecast["game_id"],
                    "play_id": forecast["play_id"],
                    "nominal": level,
                    "observed": coverage(mean, cov, y, level),
                }
            )
        if forecast["split"] == "test":
            precision.append(
                {
                    "family": forecast["family"],
                    "game_id": forecast["game_id"],
                    "play_id": forecast["play_id"],
                    "energy_128_seed1729": float(
                        energy_score(gaussian_samples(mean, cov, 128, 1729), y).mean()
                    ),
                    "energy_64_seed1730": float(
                        energy_score(gaussian_samples(mean, cov, 64, 1730), y).mean()
                    ),
                }
            )
    write_json(destination / "synthetic-calibration.json", calibration)
    write_json(destination / "synthetic-monte-carlo-audit.json", precision)
    primary = report["primary"]
    names = [r["family"] for r in primary]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        axes[0].barh(names, [r["energy_score"] for r in primary], color="#c67135")
        axes[0].set_xlabel("Play-mean marginal energy score (yards; lower better)")
        axes[0].set_title(
            f"Synthetic shifted test: {primary[0]['plays']} plays / {primary[0]['games']} games"
        )
        axes[1].barh(names, [r["marginal_coverage_80"] for r in primary], color="#327c78")
        axes[1].axvline(0.8, color="black", linestyle="--", label="Nominal 80%")
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel("Observed marginal region coverage")
        axes[1].set_title("Validation scaling did not repair test shift")
        axes[1].legend()
        fig.suptitle("Software fixture evidence — no NFL forecasting claim", fontsize=12)
        fig.savefig(destination / "synthetic-scores-coverage.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        timings = report["timing"]
        means = [
            np.mean([t["late_minus_early_energy"] for t in timings if t["family"] == n])
            for n in names
        ]
        changes = [
            np.mean([t["belief_change_nats"] for t in timings if t["family"] == n]) for n in names
        ]
        axes[0].barh(names, means, color="#c67135")
        axes[0].axvline(0, color="black", linewidth=0.8)
        axes[0].set_xlabel("Late minus early energy score (yards)")
        axes[1].barh(names, changes, color="#327c78")
        axes[1].set_xlabel("Mean symmetric marginal Gaussian KL (nats)")
        fig.suptitle("Same +1s target and -1s anchor; synthetic common cohort", fontsize=12)
        fig.savefig(destination / "synthetic-timing.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for ax, partition in zip(axes, ["validation", "test"], strict=True):
            for name in names:
                levels = [0.5, 0.8, 0.95]
                observed = [
                    np.mean(
                        [
                            r["observed"]
                            for r in calibration
                            if r["family"] == name and r["split"] == partition and r["nominal"] == p
                        ]
                    )
                    for p in levels
                ]
                ax.plot(levels, observed, marker="o", label=name)
            ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)
            ax.set(
                xlim=(0, 1),
                ylim=(0, 1),
                xlabel="Nominal marginal coverage",
                ylabel="Observed coverage",
                title=f"Synthetic {partition}",
            )
        axes[1].legend(fontsize=8)
        fig.suptitle("Validation-only variance scaling; same +1s target / -1s anchor", fontsize=12)
        fig.savefig(destination / "synthetic-calibration.png", dpi=160)
        plt.close(fig)
