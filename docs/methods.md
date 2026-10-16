# Forecasts, information and evaluation

## The prediction target

For each eligible defender, predict a two-dimensional displacement at +0.5 or +1 second after snap. Inputs end at −1, −0.5 or −0.1 seconds before snap. Every target uses the same observed −1 second anchor. An apparent improvement at a later cutoff should not come from silently changing the target reference or choosing easier plays.

The feature builder takes a two-second history on a causal 10 Hz grid. Velocities use trailing differences. It rejects missing or stale history instead of filling gaps with future observations. Players align by identifier, but those identifiers are not learned identity embeddings.

## Models and useful controls

Persistence and trailing velocity establish whether a complicated model is necessary. Geometry boosting uses snapshot/context features. Individual GRUs test history without player interaction; pooled and graph variants then test two ways of adding context. Directed edges include relative position, velocity and side. Matching capacity matters when attributing a difference to graph structure.

Gaussian models produce means and positive covariance matrices. Validation scaling adjusts the reported uncertainty. Shared mixtures use scene-level component weights, so sampled defenders share the same component rather than independently selecting different scene hypotheses. This is not unrestricted dependence between all defenders.

## Evaluation design

Whole games stay together in chronological train, validation and test blocks. A common input-eligible cohort makes cutoffs comparable. Missing future targets affect scoring masks, not whether a history is allowed into the input cohort. An opponent holdout asks a separate generalisation question.

Point error alone cannot assess a probabilistic forecast. Energy scores, log scores, coverage, area and calibration diagnostics describe different properties. Paired game-cluster bootstrap intervals account for correlated plays. They do not turn a small sample into precise evidence about every formation or team.

Ambiguity is available before reveal. Belief change compares predictions made at two cutoffs. Surprise and a failed sharp region require the realised future outcome. None directly measures intent, disguise quality or quarterback perception.

## Reproducibility

Model selection, covariance scaling and case-selection rules use validation data. Frozen manifests identify source, inputs and successful trials. Failures remain registered. If an exposed test later motivates an audit, that audit must be described separately; resetting the trial database would not restore an untouched test.

The public fixtures test implementation behaviour. The historical adapters are source code only. Historical data, fitted weights, empirical reports and per-play results are excluded from this repository; no numerical NFL performance claim is made here.
