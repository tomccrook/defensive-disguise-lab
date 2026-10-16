"""Local-only audited workbench; inputs, forecasts and realised outcomes stay separate."""

import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import joblib
import polars as pl
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from disguise_lab.audit import file_hash
from disguise_lab.contracts import Forecast, ForecastInput, Reveal
from disguise_lab.fixtures import input_from_rows, reveal_from_rows
from disguise_lab.nfl import outcome as nfl_outcome
from disguise_lab.registry import Registry
from disguise_lab.study import FAMILIES, forecast_sample


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    family: Literal[
        "persistence",
        "velocity",
        "geometry",
        "temporal",
        "pooled",
        "graph",
        "pooled_mixture",
        "graph_mixture",
    ]
    sample: ForecastInput


class RevealRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_hash: str


def create_app(run_path: Path | None = None) -> FastAPI:
    api = FastAPI(title="Defensive Forecast Lab", version="0.2.0")
    cache: OrderedDict[tuple[str, str], Forecast] = OrderedDict()
    frozen: OrderedDict[str, ForecastInput] = OrderedDict()
    bundles = {}
    state: dict[str, Any] = {}

    def load():
        if state:
            return state
        path = run_path
        if path is None:
            active = Path("artifacts/active.json")
            if not active.exists():
                raise HTTPException(
                    503,
                    detail={
                        "status": "missing_models",
                        "message": "Run disguise-lab synthetic first.",
                    },
                )
            try:
                path = Path(json.loads(active.read_text())["run_path"])
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise HTTPException(503, detail={"status": "corrupt_active_manifest"}) from error
            if not path.resolve().is_relative_to(Path("artifacts").resolve()):
                raise HTTPException(503, detail="Invalid local artifact path")
        if not (path / "report.json").exists():
            raise HTTPException(503, detail={"status": "incomplete_run"})
        try:
            loaded = {
                "path": path,
                "report": json.loads((path / "report.json").read_text()),
                "dataset": json.loads((path / "dataset.json").read_text()),
                "split": json.loads((path / "split.json").read_text()),
                "plays": json.loads((path / "plays.json").read_text()),
            }
        except (OSError, ValueError) as error:
            raise HTTPException(503, detail={"status": "corrupt_run_manifest"}) from error
        if not all(
            isinstance(loaded[k], dict) for k in ["dataset", "split", "report"]
        ) or not isinstance(loaded["plays"], list):
            raise HTTPException(503, detail={"status": "corrupt_run_manifest"})
        if loaded["dataset"].get("synthetic") is not True:
            dataset = loaded["dataset"]
            canonical = Path(dataset.get("canonical_root", ""))
            if not (
                dataset.get("m0_accepted") is True
                and dataset.get("local_replay_approved") is True
                and canonical.resolve().is_relative_to(Path("data/canonical").resolve())
                and loaded["report"].get("test_exposure") == 1
            ):
                raise HTTPException(503, detail={"status": "unapproved_dataset"})
        state.update(loaded)
        return state

    def sample(game_id, play_id, cutoff, horizon):
        s = load()
        matches = [p for p in s["plays"] if p["game_id"] == game_id and p["play_id"] == play_id]
        if not matches:
            raise HTTPException(404, detail={"status": "unknown_play"})
        tracking = (
            s["path"] / "tracking.parquet"
            if s["dataset"].get("synthetic")
            else Path(s["dataset"]["canonical_root"]) / matches[0]["partition"]
        )
        if not tracking.exists():
            raise HTTPException(503, detail={"status": "tracking_unavailable"})
        rows = (
            pl.scan_parquet(tracking)
            .filter((pl.col("game_id") == game_id) & (pl.col("play_id") == play_id))
            .collect()
        )
        try:
            value = input_from_rows(rows, matches[0], cutoff, horizon, s["dataset"]["dataset_id"])
        except ValueError as error:
            raise HTTPException(
                422, detail={"status": "invalid_sample", "message": str(error)}
            ) from error
        return value, rows

    @api.get("/api/health")
    def health():
        return {"status": "ok", "mode": "local_research", "external_replay": False}

    @api.get("/api/data")
    def data():
        s = load()
        return {
            "dataset": s["dataset"],
            "split": s["split"],
            "scope": s["report"]["scope"],
            "verified_nfl_games": s["report"]["verified_nfl_games"],
            "verified_bengals_plays": s["report"]["verified_bengals_plays"],
            "access_status": (
                "2022 weeks 1–9; all source hashes verified; private local study"
                if not s["dataset"].get("synthetic")
                else "Synthetic software fixtures; empirical scores belong to a separate run"
            ),
            "models": FAMILIES
            if s["dataset"].get("synthetic")
            else FAMILIES + ["pooled_mixture", "graph_mixture"],
            "selected_demo_family": s["report"].get("selected_demo_family", "geometry"),
            "budget": s["report"]["budget"],
        }

    @api.get("/api/plays")
    def plays(
        view: Literal["bengals_offense", "bengals_defense", "league"] = "league",
        held_out: bool = True,
    ):
        s = load()
        selected = [{**p, "split": s["split"]["games"][p["game_id"]]} for p in s["plays"]]
        if view != "league":
            field = "offense" if view == "bengals_offense" else "defense"
            selected = [p for p in selected if p[field] == "CIN"]
        if held_out:
            selected = [p for p in selected if p["split"] == "test"]
        # Deliberately no descriptions, outcome tags or case selection labels.
        public_fields = [
            "game_id",
            "play_id",
            "date",
            "season",
            "week",
            "offense",
            "defense",
            "synthetic",
            "split",
        ]
        return {
            "status": "ok" if selected else "empty_cohort",
            "plays": [{k: p[k] for k in public_fields if k in p} for p in selected],
        }

    @api.get("/api/sample/{game_id}/{play_id}", response_model=ForecastInput)
    def get_sample(
        game_id: str,
        play_id: str,
        cutoff: float = -0.1,
        horizon: float = 1.0,
    ):
        value, _ = sample(game_id, play_id, cutoff, horizon)
        return value

    @api.post("/api/forecast", response_model=Forecast)
    def forecast(request: PredictionRequest):
        s = load()
        if request.family not in bundles:
            path = s["path"] / f"{request.family}.joblib"
            if not path.exists():
                raise HTTPException(503, detail={"status": "model_unavailable"})
            expected = s["report"].get("serving_hashes", {}).get(request.family)
            if expected is not None and file_hash(path) != expected:
                raise HTTPException(503, detail={"status": "checkpoint_integrity_failed"})
            # Only locally produced trusted artifacts. Never accept uploaded pickle files.
            try:
                bundles[request.family] = joblib.load(path)
            except Exception as error:
                raise HTTPException(
                    503,
                    detail={"status": "model_unloadable", "error_type": type(error).__name__},
                ) from error
        bundle = bundles[request.family]
        key = (request.sample.input_hash(), bundle["model_id"] + bundle["calibration_id"])
        if key not in cache:
            try:
                start = time.perf_counter()
                cache[key] = forecast_sample(request.sample, bundle)
                s["last_inference_ms"] = (time.perf_counter() - start) * 1000
            except (ValueError, FloatingPointError) as error:
                raise HTTPException(
                    422, detail={"status": "numerical_or_input_error", "message": str(error)}
                ) from error
        cache.move_to_end(key)
        frozen[key[0]] = request.sample
        frozen.move_to_end(key[0])
        if len(cache) > 256:
            cache.popitem(last=False)
        if len(frozen) > 256:
            frozen.popitem(last=False)
        return cache[key]

    @api.post("/api/reveal", response_model=Reveal)
    def reveal(request: RevealRequest):
        if request.input_hash not in frozen:
            raise HTTPException(409, detail={"status": "forecast_required_before_reveal"})
        value = frozen[request.input_hash]
        _, rows = sample(value.game_id, value.play_id, value.cutoff, value.horizon)
        if not load()["dataset"].get("synthetic"):
            meta = next(
                p
                for p in load()["plays"]
                if p["game_id"] == value.game_id and p["play_id"] == value.play_id
            )
            return nfl_outcome(rows, value, meta["dead_t"])
        return reveal_from_rows(rows, value)

    @api.get("/api/metrics")
    def metrics():
        r = load()["report"]
        value = {
            k: r[k]
            for k in [
                "primary",
                "graph_minus_pooled",
                "timing",
                "scope",
                "opponent_primary",
                "opponent_graph_minus_pooled",
                "all_variants",
                "ablations",
                "validation",
                "timing_intervals",
                "robustness",
                "formation_audit",
            ]
            if k in r
        }
        correction = load()["path"] / "baseline-correction/report.json"
        if correction.exists():
            value["baseline_correction"] = json.loads(correction.read_text())
        return value

    @api.get("/api/cases")
    def cases():
        return {
            "mode": (
                "retrospective_synthetic_case_selection"
                if load()["dataset"].get("synthetic")
                else "retrospective_held_out_historical_selection"
            ),
            "synthetic": load()["dataset"].get("synthetic", False),
            "cases": load()["report"]["cases"],
        }

    @api.get("/api/figures/{name}")
    def figure(
        name: Literal["nfl-scores-coverage", "nfl-calibration", "nfl-timing", "nfl-sensitivities"],
    ):
        if load()["dataset"].get("synthetic"):
            raise HTTPException(404, detail={"status": "empirical_figure_unavailable"})
        path = Path("docs/evidence") / f"{name}.png"
        if not path.exists():
            raise HTTPException(503, detail={"status": "figure_not_exported"})
        return FileResponse(path, media_type="image/png")

    @api.get("/api/jobs")
    def jobs():
        report = load()["report"]
        return {
            "trials": Registry(Path("artifacts/trials.sqlite"), 36).records(),
            "max_trials": report["budget"].get("total_registry_trials", 24),
        }

    @api.get("/api/evidence/{name}")
    def evidence(
        name: Literal[
            "research-report",
            "bengals-cases",
            "interview",
            "data-access",
            "milestones",
            "ui-design-decisions",
            "empirical-plan",
        ],
    ):
        path = Path("docs") / f"{name}.md"
        if not path.exists():
            raise HTTPException(404, detail={"status": "report_pending"})
        return {"name": name, "markdown": path.read_text()}

    return api


app = create_app()
