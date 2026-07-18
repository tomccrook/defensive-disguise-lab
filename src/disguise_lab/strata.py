"""Annotation-only descriptive strata; never part of causal feature or inference paths."""

from collections import Counter
from pathlib import Path

import polars as pl

from disguise_lab.evaluation import aggregate
from disguise_lab.probability import clustered_interval


def formation_strata(metadata, split, primary, raw: Path):
    labels = pl.read_csv(raw / "plays.csv", null_values=["NA"], infer_schema_length=20000).select(
        "gameId", "playId", "offenseFormation"
    )
    lookup = {
        (str(r["gameId"]), str(r["playId"])): r["offenseFormation"] or "Missing"
        for r in labels.to_dicts()
    }
    counts = {
        part: Counter(
            lookup[(m["game_id"], m["play_id"])]
            for m in metadata
            if split["games"][m["game_id"]] == part
        )
        for part in ["train", "validation", "test"]
    }
    strata = []
    for formation in sorted(counts["test"]):
        selected = [r for r in primary if lookup[(r["game_id"], r["play_id"])] == formation]
        representative = [r for r in selected if r["family"] == "graph"]
        if len(representative) < 50 or len({r["game_id"] for r in representative}) < 3:
            strata.append(
                {
                    "formation": formation,
                    "status": "insufficient_descriptive_slice",
                    "plays": len(representative),
                    "games": len({r["game_id"] for r in representative}),
                }
            )
            continue
        metrics = aggregate(selected)
        graph = {(r["game_id"], r["play_id"]): r["energy_score"] for r in representative}
        pooled = {
            (r["game_id"], r["play_id"]): r["energy_score"]
            for r in selected
            if r["family"] == "pooled"
        }
        strata.append(
            {
                "formation": formation,
                "status": "descriptive",
                "train_plays": counts["train"][formation],
                "metrics": metrics,
                "graph_minus_pooled": clustered_interval(
                    [k[0] for k in graph], [graph[k] - pooled[k] for k in graph], draws=2000
                ),
            }
        )
    return {
        "counts": {k: dict(v) for k, v in counts.items()},
        "strata": strata,
        "scope": (
            "Retrospective source annotation strata; not model inputs. "
            "Conditions have different populations; no causal shift effect."
        ),
    }
