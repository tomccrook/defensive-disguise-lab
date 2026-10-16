import numpy as np
import polars as pl
import torch
from torch_geometric.data import Batch

from disguise_lab.empirical import draws, marginal_nll, moments
from disguise_lab.features import build_features
from disguise_lab.fixtures import generate, input_from_rows
from disguise_lab.models import SharedMixtureModel, mixture_loss


def test_shared_mixture_probabilities_batching_and_permutation():
    rows, metadata = generate(6, 1)
    play = rows.filter(pl.col("game_id") == metadata[0]["game_id"])
    sample = input_from_rows(play, metadata[0], -0.1, 1)
    f = build_features(sample)
    torch.manual_seed(1729)
    model = SharedMixtureModel("graph").eval()
    graph = f.graph()
    mean, cov, weights = model(graph)
    torch.testing.assert_close(weights.exp().sum(-1), torch.ones(1))
    assert torch.linalg.eigvalsh(cov).min() > 0
    combined = Batch.from_data_list([graph, graph])
    m, c, w = model(combined)
    torch.testing.assert_close(m[:11], mean)
    torch.testing.assert_close(c[11:], cov)
    torch.testing.assert_close(w, weights.repeat(2, 1))
    shuffled = sample.model_copy(update={"entities": tuple(reversed(sample.entities))})
    fm = build_features(shuffled)
    sm, sc, sw = model(fm.graph())
    torch.testing.assert_close(sm.flip(0), mean)
    torch.testing.assert_close(sc.flip(0), cov)
    torch.testing.assert_close(sw, weights)
    loss = mixture_loss(
        m,
        c,
        w,
        torch.zeros(22, 2),
        torch.ones(22, dtype=torch.bool),
        torch.arange(2).repeat_interleave(11),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_mixture_density_moments_and_shared_scene_draws():
    mean = np.zeros((1, 11, 2, 2))
    mean[:, :, 1, 0] = 10
    cov = np.broadcast_to(np.eye(2) * 0.0001, (1, 11, 2, 2, 2)).copy()
    weights = np.array([[0.25, 0.75]])
    point, variance = moments(mean, cov, weights)
    np.testing.assert_allclose(point[:, :, 0], 7.5)
    np.testing.assert_allclose(variance[:, :, 0, 0], 18.7501)
    samples = draws(mean, cov, 512, 1729, weights).reshape(512, 1, 11, 2)
    # Across a given draw defenders share the component, not independent selections.
    assert ((samples[:, :, 0, 0] > 5) == (samples[:, :, 10, 0] > 5)).all()
    density = marginal_nll(mean, cov, np.zeros((1, 11, 2)), weights)
    assert np.isfinite(density).all()
    from disguise_lab.probability import gaussian_nll

    np.testing.assert_allclose(
        density, gaussian_nll(mean[:, :, 0], cov[:, :, 0], np.zeros((1, 11, 2))) - np.log(0.25)
    )


def test_cutoff_endpoint_head_is_equivariant_to_target_reference():
    from dataclasses import replace

    from disguise_lab.models import Baseline

    rows, metadata = generate(6, 1)
    sample = input_from_rows(
        rows.filter(pl.col("game_id") == metadata[0]["game_id"]), metadata[0], -0.1, 1
    )
    f = build_features(sample)
    model = Baseline("geometry")
    model.anchor_residual = True

    # A fixed physical endpoint hypothesis must translate with target reference,
    # while its covariance is invariant. This tests the coordinate contract.
    class EndpointFixture:
        def predict(self, x):
            return np.tile([0.4, -0.2], (len(x), 1))

    model.estimator = EndpointFixture()
    first, covariance = model.predict(f)
    moved_reference = replace(f, anchor=f.anchor + [2, -3])
    second, changed_cov = model.predict(moved_reference)
    np.testing.assert_allclose(
        first + f.anchor[f.defenders], second + moved_reference.anchor[f.defenders], atol=1e-6
    )
    np.testing.assert_array_equal(covariance, changed_cov)
