"""Stable bivariate Gaussian scores, validation scaling and game-clustered evidence."""

import numpy as np
from scipy.stats import chi2


def gaussian_nll(mean: np.ndarray, covariance: np.ndarray, y: np.ndarray) -> np.ndarray:
    residual = y - mean
    sign, logdet = np.linalg.slogdet(covariance)
    if np.any(sign <= 0) or not np.isfinite(covariance).all():
        raise ValueError("positive finite covariance required")
    mahal = np.einsum("...i,...ij,...j->...", residual, np.linalg.inv(covariance), residual)
    return np.log(2 * np.pi) + logdet / 2 + mahal / 2


def gaussian_samples(mean: np.ndarray, covariance: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count < 2:
        raise ValueError("energy score requires at least two independent samples")
    z = np.random.default_rng(seed).normal(size=(count, len(mean), 2))
    return mean[None] + np.einsum("nij,mnj->mni", np.linalg.cholesky(covariance), z)


def energy_score(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = len(samples)
    if m < 2:
        raise ValueError("M >= 2 required")
    first = np.linalg.norm(samples - y[None], axis=-1).mean(axis=0)
    pair = np.linalg.norm(samples[:, None] - samples[None, :], axis=-1)
    # Diagonal distances are zero. Unbiased estimator can be negative; never clip.
    return first - pair.sum(axis=(0, 1)) / (2 * m * (m - 1))


def mahalanobis(mean: np.ndarray, covariance: np.ndarray, y: np.ndarray) -> np.ndarray:
    r = y - mean
    return np.einsum("...i,...ij,...j->...", r, np.linalg.inv(covariance), r)


def fit_scale(mean: np.ndarray, covariance: np.ndarray, y: np.ndarray) -> float:
    """NLL-optimal single variance multiplier, fitted on validation only."""
    return float(max(0.05, np.mean(mahalanobis(mean, covariance, y)) / 2))


def coverage(mean: np.ndarray, covariance: np.ndarray, y: np.ndarray, nominal: float) -> float:
    return float(np.mean(mahalanobis(mean, covariance, y) <= chi2.ppf(nominal, df=2)))


def belief_change(m0: np.ndarray, c0: np.ndarray, m1: np.ndarray, c1: np.ndarray) -> float:
    """Mean symmetric KL in nats for fixed-target marginal Gaussian beliefs."""

    def kl(ma, ca, mb, cb):
        inv = np.linalg.inv(cb)
        delta = mb - ma
        return 0.5 * (
            np.einsum("nij,nji->n", inv, ca)
            + np.einsum("ni,nij,nj->n", delta, inv, delta)
            - 2
            + np.linalg.slogdet(cb)[1]
            - np.linalg.slogdet(ca)[1]
        )

    return float((kl(m0, c0, m1, c1) + kl(m1, c1, m0, c0)).mean() / 2)


def clustered_interval(
    game_ids: list[str], differences: list[float], seed: int = 1729, draws: int = 1000
) -> dict:
    ids, values = np.array(game_ids), np.array(differences)
    unique = sorted(set(game_ids))
    if len(unique) < 2:
        return {"status": "insufficient_games", "games": len(unique), "interval": None}
    aggregates = np.array([[values[ids == g].sum(), (ids == g).sum()] for g in unique])
    choices = np.random.default_rng(seed).integers(0, len(unique), (draws, len(unique)))
    totals = aggregates[choices].sum(axis=1)
    boot = totals[:, 0] / totals[:, 1]
    return {
        "status": "descriptive_small_sample" if len(unique) < 10 else "ok",
        "games": len(unique),
        "plays": len(values),
        "difference": float(values.mean()),
        "interval": np.quantile(boot, [0.025, 0.975]).tolist(),
        "resampling_unit": "game",
        "draws": draws,
        "seed": seed,
    }
