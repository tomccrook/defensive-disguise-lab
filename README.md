# Defensive Disguise Lab

[![Tests](https://github.com/tomccrook/defensive-disguise-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/tomccrook/defensive-disguise-lab/actions/workflows/tests.yml)

**Can pre-snap movement help forecast where defenders will be after the snap?** This workbench compares simple motion rules with temporal and graph models, while keeping future information out of their inputs.

[Interactive synthetic demo](https://tomcrook.dev/demos/defensive-disguise) · [Project study](https://tomcrook.dev/projects/defensive-disguise) · [Methods](docs/methods.md)

The Python implementation includes data contracts, causal feature construction, baseline and neural forecasts, covariance calibration, whole-game evaluation and a local API. This public release runs on independently generated synthetic formations. It contains no historical tracking records, licensed-data reports or trained historical weights.

## Run a small example

Use Python 3.14.8 and [uv](https://docs.astral.sh/uv/getting-started/installation/). The dependency lock is committed.

```bash
uv sync --locked
uv run python examples/causal_features.py
uv run python -m pytest
```

The example builds a two-second pre-snap history from fictional players and prints the resulting feature dimensions. Tests generate their own fixtures. No tracking-data download or NFL account is required.

To train a small synthetic fixture and open the workbench, use Node 24.21.0:

```bash
uv run disguise-lab synthetic --epochs 2 --games 6 --plays 1
npm ci --prefix apps/web
uv run disguise-lab serve
```

In a second terminal:

```bash
npm run dev --prefix apps/web
```

Open `http://127.0.0.1:5178`; API documentation is at `http://127.0.0.1:8008/docs`. The short training run is a software demonstration, not an accuracy benchmark. New trials and outputs are written under ignored `artifacts/`.

## What the models see

A forecast receives a causal two-second history, the observed line of scrimmage and a chosen prediction cutoff. It predicts an endpoint at +0.5 or +1 second after snap. A common −1 second anchor keeps targets comparable across cutoffs. Trailing differences estimate velocity; future frames, realised coverage and play outcomes cannot enter the feature builder.

| Model | What it adds |
|---|---|
| Persistence | Hold the last observed position. A useful reference before adding complexity. |
| Trailing velocity | Extrapolate recent motion. |
| Boosted geometry | Learn a nonlinear displacement from current geometry and motion features. |
| Individual temporal | Encode each player's observed history with a GRU. |
| Pooled context | Add separate offensive and defensive scene summaries. |
| Graph context | Exchange pairwise messages using relative position, velocity and side. |
| Shared mixtures | Represent several scene-level endpoint hypotheses with shared component weights. |

The implementation evaluates endpoints and distributions. Lines joining endpoints in the viewer are visual interpolation, not continuous trajectories predicted by the models. Uncertainty, changing belief and realised surprise have separate definitions; none proves a defender's intent.

## Where the care went

- **One input path:** training and API inference use the same [`features.py`](src/disguise_lab/features.py).
- **Comparable evaluation:** whole games stay together, observations determine eligibility, and missing future targets are masked only for scoring.
- **Uncertainty that can be checked:** positive covariance, held-out calibration and distributional scores sit alongside point error.
- **Explicit reveal:** predicted endpoints and future outcomes use separate response contracts. Selecting another condition clears the revealed result.
- **Reproducible trials:** manifests, frozen choices, checksums and retained failures connect an output to its inputs.

Start with [`models.py`](src/disguise_lab/models.py), [`evaluation.py`](src/disguise_lab/evaluation.py), [`probability.py`](src/disguise_lab/probability.py), and the [design notes](docs/methods.md).

## Checks and publication scope

```bash
uv run ruff check src tests
uv run mypy src
uv run python -m pytest
npm run build --prefix apps/web
```

The tests exercise causal boundaries, rotation conventions, batching/permutation behaviour, mixture moments, calibration, source-adapter validation, corrupt manifests and forecast/reveal separation. They do not need historical data. Optional historical-source adapters are included as implementation code; using them requires a separate access and publication review. This release does not transfer any private access authorisation.

Historical study reports and their empirical metrics are deliberately absent. Synthetic reproduction tests establish software behaviour; they do not establish NFL forecasting performance. See [data policy](docs/data-policy.md) and [release notes](docs/public-release.md). There is no affiliation with the NFL or a team.
