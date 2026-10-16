"""Aggregate-only empirical exports and reproducible publication-style figures."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from disguise_lab.audit import file_hash
from disguise_lab.evaluation import aggregate, cases, paired
from disguise_lab.probability import clustered_interval
from disguise_lab.registry import Registry
from disguise_lab.study import FAMILIES, write_json


def interval_text(value):
    return (
        f"{value['difference']:+.4f} yd [{value['interval'][0]:+.4f}, {value['interval'][1]:+.4f}]"
    )


def metric_table(rows):
    header = "| Model | Energy ↓ (yd) | Mean error (yd) | NLL | Measured 80% | Games / plays |\n"
    header += "|---|---:|---:|---:|---:|---:|\n"
    return header + "\n".join(
        f"| {r['family']} | {r['energy_score']:.4f} | {r['endpoint_error']:.4f} | "
        f"{r['nll']:.4f} | {r['marginal_coverage_80']:.1%} | {r['games']} / {r['plays']} |"
        for r in rows
    )


def export_empirical(run: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    report = json.loads((run / "report.json").read_text())
    dataset = json.loads((run / "dataset.json").read_text())
    if dataset["synthetic"] or report["test_exposure"] != 1:
        raise ValueError("requires completed locked empirical study")
    config = json.loads((run / "preregistration.json").read_text())["config"]
    records = pl.read_parquet(run / "scores.parquet").to_dicts()
    primary = [
        r
        for r in records
        if r["protocol"] == "chronological"
        and r["split"] == "test"
        and r["cutoff"] == -0.1
        and r["horizon"] == 1
    ]
    absolute = {
        name: clustered_interval(
            [r["game_id"] for r in primary if r["family"] == name],
            [r["energy_score"] for r in primary if r["family"] == name],
            draws=2000,
        )
        for name in FAMILIES
    }
    conditions = {
        f"{cutoff}/{horizon}": aggregate(
            [
                r
                for r in records
                if r["protocol"] == "chronological"
                and r["split"] == "test"
                and r["cutoff"] == cutoff
                and r["horizon"] == horizon
            ]
        )
        for cutoff in [-1, -0.5, -0.1]
        for horizon in [0.5, 1]
    }
    h1 = paired(primary, "geometry", "geometry_static", config)
    temporal = paired(primary, "temporal", "temporal_static", config)
    hist = {
        "geometry_history_minus_static": h1,
        "individual_history_minus_static": temporal,
        "temporal_minus_geometry": paired(primary, "temporal", "geometry", config),
        "graph_minus_geometry": paired(primary, "graph", "geometry", config),
    }
    evidence = {k: v for k, v in report.items() if k not in ["timing", "trials"]}
    evidence.update(
        {
            "absolute_energy_intervals": absolute,
            "conditions": conditions,
            "history_controls": hist,
            "no_raw_tracking": True,
            "public_redistribution_clearance": False,
        }
    )
    write_json(destination / "nfl-results.json", evidence)
    write_json(destination / "nfl-trial-history.json", report["trials"])
    write_json(destination / "nfl-model-freeze.json", report["model_freeze"])
    write_json(
        destination / "nfl-test-exposure.json", json.loads((run / "test-exposure.json").read_text())
    )
    pl.DataFrame(report["all_variants"]).write_csv(destination / "nfl-primary-metrics.csv")
    write_json(
        destination / "nfl-export-manifest.json",
        {
            "run": run.name,
            "dataset_id": report["dataset_id"],
            "test_exposure": 1,
            "scores_sha256": file_hash(run / "scores.parquet"),
            "scope": (
                "Aggregate derived evidence; tracking, per-play distributions "
                "and private replay remain ignored"
            ),
        },
    )
    plots(report, destination, absolute)
    research(report, dataset, destination.parent, hist)
    correction_path = run / "baseline-correction" / "report.json"
    if correction_path.exists():
        correction = json.loads(correction_path.read_text())
        write_json(destination / "nfl-baseline-correction.json", correction)
        records = Registry(Path("artifacts/trials.sqlite"), 36).records()
        write_json(destination / "nfl-full-trial-history.json", records)
        corrective_trials = [
            r for r in records if r["config"].get("protocol") == "post_exposure_anchor_correction"
        ]
        write_json(
            destination / "nfl-baseline-correction-exposure.json",
            {
                "original_exposure": 1,
                "corrective_exposure": 2,
                "original_test_already_exposed": True,
                "correction_registered_before_fitting": corrective_trials[0]["created"],
                "scope": (
                    "Audit reconstructed from immutable trial timestamps after completion; "
                    "original exposure never reset."
                ),
                "test_scope": (
                    "Two fixed anchor-adjusted baseline variants only; "
                    "original neural fits, calibration and model selection unchanged."
                ),
                "trials": [r["trial_id"] for r in corrective_trials],
            },
        )
        corrective_report(correction, destination.parent)
    case_report(report, dataset, destination.parent, primary, config)


def corrective_report(correction, docs):
    path = docs / "research-report.md"
    text = path.read_text()
    history = interval_text(correction["comparisons"]["history_minus_static_residual"])
    graph = interval_text(correction["comparisons"]["graph_minus_geometry_residual"])
    text += f"""
## Separate post-exposure baseline audit

The main study above is frozen exposure 1. A subsequent audit found that the
boosted geometry heads predict full anchor-relative targets, whereas neural heads
add the known cutoff-to-anchor displacement. This also confounds the original
boosted history/static contrast: static features lack that explicit displacement.
The individual GRU history/static control shares the same endpoint reference and
remains the cleaner predeclared H1 comparison. Both original boosted models remain
valid forecasts; their contrast alone does not isolate a history effect.

Two additional controls were registered before their fitting, **after** the original
test was exposed. They fit target minus observed cutoff-to-anchor displacement and
add it back at inference. Tree capacity, iterations, seeds, source, splits,
covariance estimation and scaling procedures stay fixed. Corrective scales are
refitted on the same validation block only. No original model
was retrained, no test-selected setting was introduced, and demo selection is unchanged.

{metric_table(correction["primary"])}

The table above reports this run's fitted controls. Compare their validation and
test scores with the original frozen controls before interpreting any change.
Adjusted history−static: {history}.
Graph−adjusted history booster: {graph}.
These intervals describe a **post-exposure corrective audit**, not new untouched
confirmation. Original graph−pooled primary result and model selection are preserved.

The [full registry](evidence/nfl-full-trial-history.json) retains fixture, main-study
and corrective records separately. Its trial counts and timestamps remain part
of the audit rather than being reset after test exposure.
[Correction results](evidence/nfl-baseline-correction.json),
[exposure 2 audit](evidence/nfl-baseline-correction-exposure.json) and
[registered correction](../configs/baseline-correction.json) expose the sequence.
The correction audit's timestamps are reconstructed from immutable trial records;
the original test-exposure counter was never reset. Private per-play distributions,
validation/test scores and both corrective checkpoints remain under
`artifacts/nfl2022-v1/baseline-correction/`.
"""
    path.write_text(text)


def case_report(report, dataset, docs, primary, config):
    family = report["selected_demo_family"]
    text = (
        "# League cases and optional Bengals views\n\n"
        "These are retrospectively selected held-out 2022 cases, not fresh blind trials. "
        f"The replay model **{family}** was selected on validation only. "
        "All cases use −0.1s cutoff, +1s endpoint and −1s anchor. "
        "Rules select seeded ordinary, maximum spread, maximum NLL and maximum NLL "
        "among validation-sharp region failures. Duplicates remain if rules overlap.\n\n"
    )
    groups = [("League", primary, report["cases"])]
    for title, field in [
        ("CIN offence facing opposing defences", "offense"),
        ("CIN defending — separate labelled view", "defense"),
    ]:
        subset = [r for r in primary if r[field] == "CIN"]
        groups.append((title, subset, cases(subset, family, config) if subset else []))
    for title, rows, selected in groups:
        text += f"## {title}\n\n"
        results = aggregate([r for r in rows if r["family"] in FAMILIES or r["family"] == family])
        if not results:
            text += "No eligible held-out endpoints in this view.\n\n"
            continue
        text += metric_table(results) + "\n\n"
        text += (
            "Small team views are descriptive. Opponents, populations and target masks "
            "differ; they were not chosen to improve league results.\n\n"
            "| Rule | Game / play | Offence → defence | Spread (yd²) | NLL | "
            "Error (yd) | Sharp failed defenders |\n"
            "|---|---|---|---:|---:|---:|---:|\n"
        )
        for case in selected:
            e = case["evidence"]
            text += (
                f"| {case['kind']} | {case['game_id']} / {case['play_id']} | "
                f"{case['offense']} → {case['defense']} | "
                f"{e['ambiguity_trace_yards2']:.3f} | {e['nll']:.3f} | "
                f"{e['endpoint_error']:.3f} | {e['sharp_failed_defenders']} |\n"
            )
        text += "\n"
    text += (
        f"The full eligible dataset contains {dataset['bengals_offense_plays']} CIN "
        f"offensive and {dataset['bengals_defense_plays']} CIN defending plays across "
        "nine scheduled CIN games. The table above uses only final-block test plays. "
        "No postseason, full-season or current-season claim is made.\n\n"
        "Predictive ambiguity is known before reveal; surprise and sharp failure need "
        "the outcome. A model failure can reflect tracking/alignment error, missing "
        "context, unfamiliar formations, multimodality or calibration weakness. "
        "These labels are possible explanations, not verified causal diagnoses. "
        "Human football review has not established intent or quarterback perception.\n\n"
        "Open the loopback workbench's Held-out cases, or select the optional CIN "
        "study view and the listed game/play. Case replay resets to the registered "
        "model/condition and marks exploration. Future paths and actual targets "
        "remain hidden until explicit reveal. The moving field camera uses pre-snap "
        "LOS only. Private NFL screenshots and per-defender forecasts stay in ignored "
        "local artifacts; ordinary commits contain no tracking coordinates.\n\n"
        "[Full research](research-report.md) · [Results](evidence/nfl-results.json) · "
        "[Cohort](data-cohort-policy.md) · [Rights](data-access.md)\n"
    )
    (docs / "bengals-cases.md").write_text(text)


def plots(report, destination, absolute):
    rows = report["primary"]
    names = [r["family"] for r in rows]
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        centers = np.array([r["energy_score"] for r in rows])
        lower = np.array([absolute[n]["interval"][0] for n in names])
        upper = np.array([absolute[n]["interval"][1] for n in names])
        axes[0].barh(names, centers, color="#c67135")
        axes[0].errorbar(
            centers,
            names,
            xerr=np.stack([centers - lower, upper - centers]),
            fmt="none",
            color="black",
            capsize=3,
        )
        axes[0].set(
            xlabel="Marginal energy score (yd; lower better)",
            title="95% game-cluster bootstrap intervals",
        )
        axes[1].barh(names, [r["marginal_coverage_80"] for r in rows], color="#327c78")
        axes[1].axvline(0.8, color="black", linestyle="--", label="Nominal 80%")
        axes[1].set(
            xlim=(0, 1),
            xlabel="Measured marginal coverage",
            title="Validation scaling applied once",
        )
        axes[1].legend()
        fig.suptitle("2022 weeks 8–9: 28 held-out games · -0.1s → +1s · common -1s anchor")
        fig.savefig(destination / "nfl-scores-coverage.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for ax, part in zip(axes, ["validation", "test"], strict=True):
            source = report["validation"] if part == "validation" else report["all_variants"]
            for name in names + ["graph_mixture", "pooled_mixture"]:
                row = next(r for r in source if r["family"] == name)
                levels = [50, 80, 90, 95]
                ax.plot(
                    np.array(levels) / 100,
                    [row[f"marginal_coverage_{q}"] for q in levels],
                    marker=".",
                    label=name,
                )
            ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)
            ax.set(
                xlim=(0, 1),
                ylim=(0, 1),
                xlabel="Nominal marginal region",
                ylabel="Measured coverage",
                title=part,
            )
        axes[1].legend(fontsize=7)
        fig.suptitle(
            "Same target; Gaussian ellipses / sampled mixture density sets; no joint 80% claim"
        )
        fig.savefig(destination / "nfl-calibration.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        deltas = [report["timing_intervals"][n]["difference"] for n in names]
        axes[0].barh(names, deltas, color="#c67135")
        axes[0].axvline(0, color="black", linewidth=0.8)
        axes[0].set(
            xlabel="Late minus early energy score (yd)",
            title="-1s versus -0.1s\nSame +1s physical target",
        )
        axes[1].barh(
            names,
            [
                np.mean([t["belief_change_nats"] for t in report["timing"] if t["family"] == n])
                for n in names
            ],
            color="#327c78",
        )
        axes[1].set(
            xlabel="Mean symmetric marginal Gaussian KL (nats)",
            title="Belief update size\nDoes not imply improvement",
        )
        fig.suptitle("Common input cohort and fixed anchor; retrospective snap conditioning")
        fig.savefig(destination / "nfl-timing.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        checks = report["robustness"]["stress"]
        labels = [f"{r['family']} / {r['stress']}" for r in checks if r["stress"] != "unmodified"]
        values = [
            r["change_from_unmodified"]["difference"] for r in checks if r["stress"] != "unmodified"
        ]
        ax.barh(labels, values, color="#327c78")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(
            xlabel="Change in energy score (yd); models and calibration frozen",
            title="Predeclared 256-play sensitivity subset; descriptive",
        )
        fig.savefig(destination / "nfl-sensitivities.png", dpi=160)
        plt.close(fig)


def research(report, dataset, docs, hist):
    conditions = "−0.1s cutoff, +1.0s target from common −1.0s anchor"
    selected = report["selected_demo_family"]
    main = metric_table(report["primary"])
    controls = "\n".join(
        f"- {k.replace('_', ' ')}: {interval_text(v)}." for k, v in report["ablations"].items()
    )
    timing = "\n".join(
        f"- {name}: {interval_text(report['timing_intervals'][name])}." for name in FAMILIES
    )
    before = [
        r
        for r in report["validation_calibration"]
        if r["protocol"] == "chronological"
        and r["cutoff"] == -0.1
        and r["horizon"] == 1
        and r["family"] in FAMILIES
    ]
    calibration = "\n".join(
        f"| {r['family']} | {r['nll_before']:.3f} → {r['nll_after']:.3f} | "
        f"{r['coverage_80_before']:.1%} → {r['coverage_80_after']:.1%} | {r['scale']:.3f} |"
        for r in before
    )
    mc = report["robustness"]["monte_carlo_precision"]
    precision = "\n".join(
        f"- {count} draws: graph−pooled range "
        f"{min(r['graph_minus_pooled'] for r in mc if r['draws'] == count):+.4f} to "
        f"{max(r['graph_minus_pooled'] for r in mc if r['draws'] == count):+.4f} yd "
        "over four fixed seeds."
        for count in [64, 128]
    )
    text = f"""# Locked historical NFL forecast study

This completed offline study uses hash-verified 2022 tracking from the recovered
2025 Big Data Bowl release: {dataset["games"]} games, weeks 1–9, {dataset["plays"]:,}
input-eligible plays. The primary population is league-wide. Results concern early
defender endpoints from idealised tracking; they do not identify defensive intent,
quarterback perception, live forecasting ability or coaching value.

Chronology is 80 train / 28 validation / 28 test whole games, 8,980 / 3,183 / 3,144
plays. Weeks 8–9 were scored once after both protocols, model selection and
validation-only calibration froze. Default demo model: **{selected}**, selected by
validation marginal energy only. The experiment registry retains all synthetic
and empirical trials, full epoch histories and checkpoint hashes.

{conditions}. Lower scores are better. Defenders are averaged within each play,
then plays equally; entire games are bootstrap clusters. These are finite-period
estimates, not multi-season evidence. Density scores use two-dimensional yards.
Games are the resampling units; repeated teams and a short calendar block may
still create dependence between games. Bootstrap intervals do not establish
independence or power for every comparison/subgroup.

{main}

![Scores and coverage](evidence/nfl-scores-coverage.png)

The predeclared primary graph-minus-capacity-matched-pooled difference is
**{interval_text(report["graph_minus_pooled"])}**, 95% percentile game bootstrap,
2,000 resamples of 28 games. Negative favours graph; an interval crossing zero
is inconclusive. Two seeds do not establish general optimisation stability.
A graph win against pooled alone does not establish superiority over every baseline.

## Hypotheses and controlled comparisons

H1, recent history versus static cutoff geometry: boosted geometry history−static
{interval_text(hist["geometry_history_minus_static"])}; individual GRU history−static
{interval_text(hist["individual_history_minus_static"])}. Same targets and cohort.
Geometry contains causal trailing velocities and pooled summaries; the static
control removes velocities. The GRU static control sees one position frame.
The original cross-family temporal−geometry comparison is
{interval_text(hist["temporal_minus_geometry"])}; graph−geometry is
{interval_text(hist["graph_minus_geometry"])}. These compare different model classes,
so neither isolates a history/edge effect by itself. Secondary intervals are
descriptive, without a multiplicity correction; H2 remains the single primary contrast.

H2 tests explicit edges after matching temporal encoder and approximate parameter
capacity (pooled 9,403, graph 9,269). IDs align players but are never embeddings or
features. Two dense directed layers use only allowed relative positions, velocities
and side relations. Additional registered comparisons:

{controls}

Mixture heads share one three-way scene component across defenders, then assume
conditional independence. They are a constrained joint approximation; weights
are not football scheme labels. Their matched pooled control has the same head.
Full variant metrics and parameter counts are in the linked registry/results.

H3, later information: late−early energy, with identical +1s endpoint and −1s
anchor on the common input-eligible cohort:

{timing}

![Timing](evidence/nfl-timing.png)

Symmetric marginal Gaussian KL measures belief change. It needs no realised outcome;
NLL and density improvement do. Mixture KL is not substituted with Gaussian moment
KL; timing KL here applies to the six Gaussian families only. Exceptions and
per-play records remain in ignored local `scores.parquet` and `report.json`.

H4 uses a separate opponent holdout: ARI/SEA whole games test (16 / 1,747 plays),
BUF/SF validation after test priority (15 / 1,635), remaining 105 / 11,925 train.
No game involving held-out teams enters that protocol's train population. It uses
all nine weeks and is **not simultaneously chronological**. Populations and training
sizes differ from chronology; overlapping games are not independent replication
and a protocol difference is not a causal shift effect.

{metric_table(report["opponent_primary"])}

Opponent graph−pooled: {interval_text(report["opponent_graph_minus_pooled"])}.
Formation counts and qualifying descriptive slices use source annotations only,
never model inputs. Some formations have too few plays/games for claims; no
formation-specific model is selected after inspecting these strata.

## Calibration, ambiguity and failure

Before/after calibration on the same validation block (fitting diagnostics, not
independent calibration evidence):

| Model | NLL before → after | 80% coverage before → after | Covariance multiplier |
|---|---:|---:|---:|
{calibration}

![Calibration](evidence/nfl-calibration.png)

Gaussian covariance multipliers minimise validation NLL separately by cutoff/horizon.
The test curves show what transferred. Mixture coverage uses approximate marginal
highest-density sets with 256 samples; Canvas component ellipses are conditional
regions, not whole-mixture 80% sets. The main ledger's mixture mean/covariance are
moment summaries. Joint energy and all-eleven marginal-region inclusion are reported
on complete target plays only; the latter is not a nominal joint 80% claim.

H5 keeps predictive spread (covariance trace), belief change (symmetric KL), realised
surprise (NLL), and sharp failure distinct. Sharp means at or below a validation
25th-percentile trace threshold for that model/condition; failure means outside its
nominal marginal 80% region. Correlations and top-decile overlap in the results
are descriptive and show which groups overlap. No one-number disguise ranking is made.

Retrospective cases use seeded ordinary play, maximum spread, maximum NLL and maximum
NLL among qualifying sharp failures, with duplicates retained if rules select the
same play. Selection tells a story about a forecast; it cannot estimate overall skill.
A confidently wrong model may lack context, be poorly calibrated, encounter a rare
formation or use imperfect tracking. No case is proof of deliberate deception.

## Sensitivity and numerical precision

![Sensitivities](evidence/nfl-sensitivities.png)

Predeclared 256-play subset: sensor bias of 0.1 yd translates each player's allowed
history and anchor consistently, with spatial features recomputed and predictions
mapped to the original physical target. Six-frame GRU stress retains the observed
anchor and is not the accepted production input schema. Graph radii 10/20 yd are
scored without recalibration or threshold selection. Production frame gaps fail
closed; causal trailing carry-forward never backfills from future observations.

{precision}

Monte Carlo precision uses a fixed 128-play subset; its sampling variability is
separate from game-sampling uncertainty. Energy's unbiased estimator is never clipped.

## Source, reproducibility and limits

[NFL/PFF original competition terms](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/rules)
declare CC BY-NC 4.0; source revision
`e28d3b25725799ef2213680016db5325f6f90014` from the specifically authorised
[recovered copy](https://github.com/b00nk3r/NFL-Big-Data-Bowl-2025) was verified by
full SHA256 and byte counts for all thirteen files. Its Apache code license does
not replace data terms. Official Kaggle inventory currently exposes a removal README.
Hashes establish integrity of the recovered copy; inaccessible official payload
fingerprints prevent an independent byte-for-byte original-release match. Attribution
rests on the source repository and compatible actual schemas/season inventory.
No raw tracking, checkpoints or private NFL screenshots are committed or published.

817 of 16,124 source plays were excluded under the recorded input/cohort rules.
Histories require 22 players and two causal seconds at all cutoffs. Outcome availability
never determines input eligibility; missing, off-field and after-ball-dead targets
are masked. The full dataset retains {dataset["observed_defender_targets_by_horizon"][0]:,}
observed defender targets at +0.5s and {dataset["observed_defender_targets_by_horizon"][1]:,}
at +1s. All models score common masks. Censoring makes this an endpoint-available
population. QB kneels/spikes/nullified plays are named retrospective exclusions.

Provider smoothing/measurement error, omitted ball/body/assignment signals, perfect
visibility, retrospective snap alignment, a partial single season, selection policy
and only two neural seeds limit inference. Side is team membership, not a privileged
assignment annotation. Realised coverage, rushers, routes and pressure are audited
separately and absent from input schemas, causal geometry, edges and model features.
No annotation classifier, learned trajectory, live feed or NFL affiliation is claimed.

Measured M4 CPU training uses four threads; complete MPS workload passes but CPU was
faster for this small graph. Windows/RTX execution remains unverified. Local raw source
is 8.17 GB and canonical/cache 599 MB; no paid cloud resources. Runtime locks, causal
mutation/permutation/batching tests, browser journeys, manifests and preserved trial
histories make the claims inspectable. Training began from source snapshot `e82ae39`;
later API/evaluation commits did not change loaded training code or numeric caches.

Evidence: [aggregate results](evidence/nfl-results.json),
[trial registry](evidence/nfl-trial-history.json),
[model freeze](evidence/nfl-model-freeze.json), [test exposure](evidence/nfl-test-exposure.json),
[data audit](evidence/nfl-quality-audit.json), [preregistration](evidence/nfl-preregistration.json),
[plan](empirical-plan.md), [case report](bengals-cases.md), [verification](verification.md).
The earlier synthetic fixture findings remain in `evidence/synthetic-*`; they are
software controls and are never pooled with NFL evidence.
"""
    (docs / "research-report.md").write_text(text)
