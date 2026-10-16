import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from disguise_lab.api import create_app
from disguise_lab.fixtures import input_from_rows
from disguise_lab.study import forecast_sample


@pytest.fixture
def active():
    active_file = Path("artifacts/active.json")
    if not active_file.exists():
        pytest.skip("run bounded synthetic smoke before integration tests")
    return Path(json.loads(active_file.read_text())["run_path"])


def test_api_forecast_reveal_boundary_and_statuses(active):
    client = TestClient(create_app(active))
    play = client.get("/api/plays").json()["plays"][0]
    key = f"{play['game_id']}/{play['play_id']}"
    sample = client.get(f"/api/sample/{key}?cutoff=-0.1&horizon=1.0").json()
    assert "future" not in sample and "outcomes" not in sample
    assert all(p["t"] <= -0.1 for e in sample["entities"] for p in e["observations"])
    assert client.post("/api/reveal", json={"input_hash": "not-a-forecast"}).status_code == 409
    forecast = client.post("/api/forecast", json={"family": "geometry", "sample": sample})
    assert forecast.status_code == 200
    body = forecast.json()
    assert not any(k in body for k in ["future", "outcomes", "surprise", "coverage_label"])
    assert (
        client.post("/api/forecast", json={"family": "geometry", "sample": sample}).json() == body
    )
    outcome = client.post("/api/reveal", json={"input_hash": body["input_hash"]})
    assert outcome.status_code == 200 and len(outcome.json()["outcomes"]) == 11
    sample["realised_coverage"] = "MAN"
    assert (
        client.post("/api/forecast", json={"family": "geometry", "sample": sample}).status_code
        == 422
    )
    assert client.get("/api/sample/unknown/unknown").status_code == 404
    assert client.get(f"/api/sample/{key}?cutoff=0.0").status_code == 422


def test_outcome_mutation_preserves_trained_forecasts(active):
    import joblib

    metadata = json.loads((active / "plays.json").read_text())[0]
    manifest = json.loads((active / "dataset.json").read_text())
    dataset = manifest["dataset_id"]
    tracking = (
        active / "tracking.parquet"
        if manifest["synthetic"]
        else Path(manifest["canonical_root"]) / metadata["partition"]
    )
    rows = pl.read_parquet(tracking).filter(
        (pl.col("game_id") == metadata["game_id"]) & (pl.col("play_id") == metadata["play_id"])
    )
    changed = rows.with_columns(
        pl.when(pl.col("t") > -0.1).then(1).otherwise(pl.col("x")).alias("x"),
        pl.lit("mutated annotation").alias("pressure"),
    )
    first = input_from_rows(rows, metadata, -0.1, 1.0, dataset)
    second = input_from_rows(changed, metadata, -0.1, 1.0, dataset)
    for family in TestClient(create_app(active)).get("/api/data").json()["models"]:
        bundle = joblib.load(active / f"{family}.joblib")
        assert forecast_sample(first, bundle) == forecast_sample(second, bundle)


def test_manifest_and_training_inference_parity(active):
    client = TestClient(create_app(active))
    if not client.get("/api/data").json()["dataset"]["synthetic"]:
        from disguise_lab.empirical import CachedNFL

        manifest = json.loads((active / "dataset.json").read_text())
        cached = CachedNFL(Path(manifest["canonical_root"]))
        arrays = np.load(active / "chronological/graph-test-distributions.npz")
        p, c, h = arrays["combos"][0]
        meta = cached.metadata[p]
        sample = client.get(
            f"/api/sample/{meta['game_id']}/{meta['play_id']}?cutoff=-1&horizon=0.5"
        ).json()
        result = client.post("/api/forecast", json={"family": "graph", "sample": sample}).json()
        mapping = {d["entity_id"]: d for d in result["defenders"]}
        np.testing.assert_allclose(
            [mapping[identity]["mean_displacement"] for identity in meta["entity_ids"][11:]],
            arrays["mean"][0],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            [mapping[identity]["covariance"] for identity in meta["entity_ids"][11:]],
            arrays["covariance"][0],
            atol=1e-5,
        )
        assert client.get("/api/plays").json()["plays"][0]["split"] == "test"
        return
    forecast_records = json.loads((active / "forecasts.json").read_text())
    record = next(r for r in forecast_records if r["family"] == "graph" and r["cutoff"] == -0.1)
    key = f"{record['game_id']}/{record['play_id']}"
    sample = client.get(
        f"/api/sample/{key}?cutoff={record['cutoff']}&horizon={record['horizon']}"
    ).json()
    result = client.post("/api/forecast", json={"family": "graph", "sample": sample}).json()
    np.testing.assert_allclose(
        [d["mean_displacement"] for d in result["defenders"]], record["mean"], rtol=1e-5
    )
    split = json.loads((active / "split.json").read_text())
    cohort = json.loads((active / "cohort.json").read_text())
    assert all(s["split"] == split["games"][s["game"]] for s in cohort["sample_ids"])


def test_missing_corrupt_unapproved_and_unloadable_artifacts(active, tmp_path, monkeypatch):
    import joblib

    empty = TestClient(create_app(tmp_path))
    assert empty.get("/api/data").json()["detail"]["status"] == "incomplete_run"
    for name in ["report", "dataset", "split", "plays"]:
        (tmp_path / f"{name}.json").write_text((active / f"{name}.json").read_text())
    (tmp_path / "dataset.json").write_text("broken JSON")
    corrupt = TestClient(create_app(tmp_path)).get("/api/data")
    assert corrupt.status_code == 503
    assert corrupt.json()["detail"]["status"] == "corrupt_run_manifest"
    dataset = json.loads((active / "dataset.json").read_text())
    dataset["synthetic"] = False
    dataset["local_replay_approved"] = False
    (tmp_path / "dataset.json").write_text(json.dumps(dataset))
    assert (
        TestClient(create_app(tmp_path)).get("/api/data").json()["detail"]["status"]
        == "unapproved_dataset"
    )
    client = TestClient(create_app(active))
    play = client.get("/api/plays").json()["plays"][0]
    sample = client.get(f"/api/sample/{play['game_id']}/{play['play_id']}").json()

    def incompatible(path):
        raise RuntimeError("fixture incompatibility")

    monkeypatch.setattr(joblib, "load", incompatible)
    failed = client.post("/api/forecast", json={"family": "graph", "sample": sample})
    assert failed.status_code == 503
    assert failed.json()["detail"]["status"] == "model_unloadable"
