import numpy as np
import polars as pl
import pytest
import torch
from torch_geometric.data import Batch

from disguise_lab.features import build_features
from disguise_lab.fixtures import generate, input_from_rows
from disguise_lab.models import MovementModel, matched_width, parameter_count
from disguise_lab.probability import (
    belief_change,
    clustered_interval,
    coverage,
    energy_score,
    fit_scale,
    gaussian_nll,
)


def test_graph_equivariance_batch_independence_and_gradients():
    torch.manual_seed(1729)
    rows, meta = generate(6, 1)
    one = rows.filter(pl.col("game_id") == meta[0]["game_id"])
    sample = input_from_rows(one, meta[0], -0.1, 1.0)
    p = np.random.default_rng(5).permutation(22)
    perm = sample.model_copy(update={"entities": tuple(sample.entities[i] for i in p)})
    f, fp = build_features(sample), build_features(perm)
    for family in ["temporal", "pooled", "graph"]:
        model = MovementModel(family)
        mean, cov = model(f.graph())
        other, _ = model(fp.graph())
        old_ids = np.array(f.ids)[f.defenders]
        new_ids = np.array(fp.ids)[fp.defenders]
        reindex = [list(old_ids).index(e) for e in new_ids]
        torch.testing.assert_close(other, mean[reindex], atol=1e-5, rtol=1e-5)
        batched, _ = model(Batch.from_data_list([f.graph(), fp.graph()]))
        torch.testing.assert_close(batched[:11], mean, atol=1e-5, rtol=1e-5)
        torch.linalg.cholesky(cov)
        (mean.square().mean() + cov.mean()).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    graph_count = parameter_count(MovementModel("graph", 24))
    pooled_count = parameter_count(MovementModel("pooled", matched_width()))
    assert abs(graph_count - pooled_count) / graph_count < 0.05


def test_scoring_known_cases_mask_and_continuous_density():
    samples = np.array([[[-1.0, 0.0]], [[1.0, 0.0]]])
    # Unbiased finite-sample ES at zero is zero, at x=-1 it is also zero.
    np.testing.assert_allclose(energy_score(samples, np.zeros((1, 2))), [0])
    with pytest.raises(ValueError):
        energy_score(samples[:1], np.zeros((1, 2)))
    cov = np.array([np.eye(2) * 0.01])
    assert gaussian_nll(np.zeros((1, 2)), cov, np.zeros((1, 2)))[0] < 0
    assert belief_change(np.zeros((1, 2)), cov, np.zeros((1, 2)), cov) == pytest.approx(0)
    assert coverage(np.zeros((1, 2)), cov, np.zeros((1, 2)), 0.8) == 1
    assert fit_scale(np.zeros((1, 2)), np.array([np.eye(2)]), np.array([[2.0, 0.0]])) == 2


def test_clustered_uncertainty():
    result = clustered_interval(["a", "a", "b", "b"], [1, 1, 3, 3])
    assert result["games"] == 2
    assert result["difference"] == 2
    assert result["interval"] == [1, 3]
    assert clustered_interval(["a"], [1])["interval"] is None
