import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from torch_geometric.data import Batch

from disguise_lab.api import create_app
from disguise_lab.contracts import ForecastInput
from disguise_lab.features import build_features, rotate, wrap_angle
from disguise_lab.fixtures import generate, input_from_rows, reveal_from_rows


@pytest.fixture
def fixture():
    rows, metadata = generate(6, 1)
    meta = metadata[0]
    rows = rows.filter(pl.col("game_id") == meta["game_id"])
    return rows, meta, input_from_rows(rows, meta, -0.1, 1.0)


def test_geometry():
    xy = np.array([[30, 10], [33, 14]])
    np.testing.assert_allclose(rotate(xy, True)[0], [90, 43.333333333333])
    np.testing.assert_allclose(rotate(rotate(xy, True), True), xy)
    assert np.linalg.norm(np.diff(rotate(xy, True), axis=0)) == pytest.approx(5)
    np.testing.assert_allclose(rotate(np.array([2, -1]), True, vector=True), [-2, 1])
    assert wrap_angle(361) == 1


def test_outcome_mutation(fixture):
    rows, meta, sample = fixture
    changed = rows.with_columns(
        pl.when(pl.col("t") > -0.1).then(100).otherwise(pl.col("x")).alias("x"),
        pl.lit("fake realised coverage").alias("pff_passCoverage"),
    )
    other = input_from_rows(changed, meta, -0.1, 1.0)
    assert sample.input_hash() == other.input_hash()
    np.testing.assert_array_equal(build_features(sample).history, build_features(other).history)


def test_input_rejects_outcomes_future_duplicates_and_gaps(fixture):
    _, _, sample = fixture
    data = sample.model_dump()
    data["coverage"] = "MAN"
    with pytest.raises(ValidationError):
        ForecastInput.model_validate(data)
    data = sample.model_dump()
    data["entities"][0]["observations"][-1]["t"] = 0.0
    with pytest.raises(ValidationError):
        ForecastInput.model_validate(data)
    data = sample.model_dump()
    data["entities"][0]["observations"] = data["entities"][0]["observations"][::3]
    with pytest.raises(ValidationError):
        ForecastInput.model_validate(data)


def test_fixed_anchor_target_and_missing_mask(fixture):
    rows, meta, late = fixture
    early = input_from_rows(rows, meta, -1.0, 1.0)
    assert reveal_from_rows(rows, early).outcomes == reveal_from_rows(rows, late).outcomes
    censored = rows.filter(~((pl.col("entity_id") == "D01") & (pl.col("t") == 1.0)))
    assert reveal_from_rows(censored, late).outcomes[0].displacement is None


def test_truncated_history_is_rejected_before_feature_construction(fixture):
    _, _, sample = fixture
    data = sample.model_dump()
    data["entities"][0]["observations"] = data["entities"][0]["observations"][1:]
    with pytest.raises(ValidationError, match="complete two-second history"):
        ForecastInput.model_validate(data)
    response = TestClient(create_app()).post(
        "/api/forecast", json={"family": "geometry", "sample": data}
    )
    assert response.status_code == 422
    assert "complete two-second history" in response.text


def test_edges_batch_and_storage_permutation(fixture):
    _, _, sample = fixture
    f = build_features(sample)
    assert f.edge_index.shape == (2, 462)
    batch = Batch.from_data_list([f.graph(), f.graph()])
    assert (batch.batch[batch.edge_index[0]] == batch.batch[batch.edge_index[1]]).all()
    p = np.random.default_rng(10).permutation(22)
    permuted = sample.model_copy(update={"entities": tuple(sample.entities[i] for i in p)})
    np.testing.assert_array_equal(build_features(permuted).history, f.history[p])
    assert permuted.input_hash() == sample.input_hash()
    send, receive = f.edge_index
    for k in range(462):
        reverse = np.where((send == receive[k]) & (receive == send[k]))[0][0]
        np.testing.assert_allclose(f.edge_attr[k, :2], -f.edge_attr[reverse, :2])
