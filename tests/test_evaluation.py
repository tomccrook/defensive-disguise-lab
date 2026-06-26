from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from disguise_lab.contracts import DefenderForecast, DefenderMode
from disguise_lab.evaluation import paired, score_predictions
from disguise_lab.features import build_features, geometry_matrix
from disguise_lab.fixtures import generate, input_from_rows
from disguise_lab.robustness import shifted_features


def fake_data():
    mask = np.ones((2, 2, 11), dtype=bool)
    mask[1, 1, 0] = False
    y = np.zeros((2, 2, 11, 2))
    y[1, 1, 0] = 999  # Missing values must not change any proper score.
    return SimpleNamespace(
        y=y,
        mask=mask,
        metadata=[
            {"game_id": str(i), "play_id": "2", "offense": "CIN", "defense": "PIT"}
            for i in range(2)
        ],
    )


def test_mixture_regions_scores_and_common_mask():
    data = fake_data()
    combos = np.array([[0, 2, 1], [1, 2, 1]])
    mean = np.zeros((2, 11, 3, 2))
    cov = np.broadcast_to(np.eye(2), (2, 11, 3, 2, 2)).copy()
    weight = np.ones((2, 3)) / 3
    bundle = {"variant": "graph_mixture", "model_id": "fixture", "sharp_thresholds": {(2, 1): 2.0}}
    cfg = {"seed": 1729, "budget": {"energy_samples": 64, "bootstrap_games": 100}}
    records, point, variance = score_predictions(
        data, combos, mean, cov, weight, bundle, cfg, "fixture", "test"
    )
    assert [r["targets"] for r in records] == [11, 10]
    assert all(r["marginal_coverage_80"] == 1 for r in records)
    assert records[1]["joint_energy_score"] is None
    assert records[0]["nll"] == pytest.approx(np.log(2 * np.pi))
    assert np.isfinite(point).all() and np.isfinite(variance).all()
    changed = fake_data()
    changed.y[1, 1, 0] = -999
    second, _, _ = score_predictions(
        changed, combos, mean, cov, weight, bundle, cfg, "fixture", "test"
    )
    assert records == second
    with pytest.raises(ValueError, match="probabilities"):
        score_predictions(data, combos, mean, cov, weight * 2, bundle, cfg, "fixture", "test")
    others = [{**r, "family": "pooled"} for r in records]
    assert paired(records + others, "graph_mixture", "pooled", cfg)["difference"] == 0
    with pytest.raises(ValueError, match="cohorts"):
        paired(records + others[:1], "graph_mixture", "pooled", cfg)


def test_coordinate_bias_updates_geometry_and_original_target():
    rows, metadata = generate(6, 1)
    import polars as pl

    sample = input_from_rows(
        rows.filter(pl.col("game_id") == metadata[0]["game_id"]), metadata[0], -0.1, 1
    )
    f = build_features(sample)
    offset = np.random.default_rng(1).normal(0, 0.1, (22, 2)).astype(np.float32)
    shifted = shifted_features(f, offset)
    np.testing.assert_allclose(shifted.last - f.last, shifted.anchor - f.anchor, atol=1e-5)
    np.testing.assert_array_equal(shifted.velocity, f.velocity)
    np.testing.assert_allclose(
        shifted.history[:, :, :2] - f.history[:, :, :2],
        np.broadcast_to(offset[:, None] / 10, (22, 21, 2)),
        atol=1e-6,
    )
    assert not np.array_equal(geometry_matrix(shifted), geometry_matrix(f))
    np.testing.assert_allclose(
        (shifted.last - shifted.anchor) + (shifted.anchor - f.anchor),
        shifted.last - f.anchor,
        atol=1e-5,
    )
    assert shifted.edge_index.shape == f.edge_index.shape
    # A production history gap fails closed; no invented future backfill.
    entity = sample.entities[0]
    gap = entity.model_copy(
        update={"observations": entity.observations[:3] + entity.observations[5:]}
    )
    with pytest.raises(ValidationError, match="gap"):
        sample.model_validate(
            {**sample.model_dump(), "entities": [gap] + list(sample.entities[1:])}
        )


def test_output_covariance_and_mixture_weights_fail_closed():
    base = dict(
        entity_id="D1", anchor=(30, 20), mean_displacement=(0, 0), covariance=((1, 0), (0, 1))
    )
    with pytest.raises(ValidationError, match="positive definite"):
        DefenderForecast(**{**base, "covariance": ((1, 2), (2, 1))})
    mode = DefenderMode(weight=0.3, mean_displacement=(0, 0), covariance=((1, 0), (0, 1)))
    with pytest.raises(ValidationError, match="sum to one"):
        DefenderForecast(**base, modes=(mode, mode))
