"""Audited NFL adapter: retrospective clock alignment, input-only causal histories."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from disguise_lab.audit import file_hash, unique_keys
from disguise_lab.contracts import WIDTH, ForecastInput, Reveal, digest
from disguise_lab.features import build_features
from disguise_lab.fixtures import input_from_rows, reveal_from_rows
from disguise_lab.study import write_json

RAW_COLUMNS = [
    "gameId",
    "playId",
    "nflId",
    "frameId",
    "frameType",
    "time",
    "club",
    "playDirection",
    "x",
    "y",
    "event",
]
DEAD_EVENTS = [
    "tackle",
    "out_of_bounds",
    "pass_outcome_incomplete",
    "touchdown",
    "qb_slide",
    "qb_kneel",
    "qb_spike",
    "fumble_out_of_bounds",
    "safety",
    "end_of_play",
]


def canonical_week(
    raw: Path, plays: pl.DataFrame, games: pl.DataFrame
) -> tuple[pl.DataFrame, list[dict], dict]:
    """Whole-week projection; outcome fields never enter ForecastInput."""
    schema = pl.scan_csv(raw, null_values=["NA"]).collect_schema()
    if not set(RAW_COLUMNS).issubset(schema):
        raise ValueError("incompatible tracking schema")
    t = (
        pl.scan_csv(raw, null_values=["NA"], schema_overrides={"nflId": pl.Int64})
        .select(RAW_COLUMNS)
        .collect(engine="streaming")
        .with_columns(pl.col("time").str.to_datetime(time_zone="UTC").alias("clock"))
    )
    unique_keys(t, ["gameId", "playId", "nflId", "frameId"], "source tracking")
    snaps = (
        t.filter((pl.col("event") == "ball_snap") | (pl.col("frameType") == "SNAP"))
        .group_by("gameId", "playId")
        .agg(
            pl.col("clock").n_unique().alias("nclock"),
            pl.col("frameId").n_unique().alias("nframe"),
            pl.col("clock").first().alias("snap"),
        )
    )
    valid = snaps.filter((pl.col("nclock") == 1) & (pl.col("nframe") == 1))
    aligned = t.join(
        valid.select("gameId", "playId", "snap"), on=["gameId", "playId"], validate="m:1"
    )
    aligned = aligned.with_columns(
        ((pl.col("clock") - pl.col("snap")).dt.total_milliseconds() / 1000).alias("t")
    )
    dead = (
        aligned.filter(pl.col("event").is_in(DEAD_EVENTS) & (pl.col("t") >= 0))
        .group_by("gameId", "playId")
        .agg(pl.col("t").min().alias("dead_t"))
    )
    # LOS is an observation of the ball at the common PRE-SNAP anchor.
    ball = (
        aligned.filter(pl.col("nflId").is_null() & ((pl.col("t") + 1).abs() < 1e-6))
        .group_by("gameId", "playId")
        .agg(pl.col("x").first().alias("ball_x"), pl.len().alias("ball_anchor_rows"))
    )
    directions = (
        aligned.filter(pl.col("t") <= -0.1)
        .group_by("gameId", "playId")
        .agg(
            pl.col("playDirection").n_unique().alias("direction_count"),
            pl.col("playDirection").first().alias("direction"),
        )
    )
    meta = (
        aligned.select("gameId", "playId")
        .unique()
        .join(plays, on=["gameId", "playId"], how="left", validate="1:1")
        .join(games, on="gameId", how="left", validate="m:1")
        .join(ball, on=["gameId", "playId"], how="left", validate="1:1")
        .join(dead, on=["gameId", "playId"], how="left", validate="1:1")
        .join(directions, on=["gameId", "playId"], how="left", validate="1:1")
    )
    if meta["possessionTeam"].null_count() or meta["season"].null_count():
        raise ValueError("source tracking has orphan play/game metadata")
    window = aligned.filter(pl.col("nflId").is_not_null() & pl.col("t").is_between(-3, 1))
    window = window.join(
        plays.select("gameId", "playId", "possessionTeam", "defensiveTeam"),
        on=["gameId", "playId"],
        validate="m:1",
    )
    bad_team = window.filter(
        (pl.col("club") != pl.col("possessionTeam")) & (pl.col("club") != pl.col("defensiveTeam"))
    )
    if bad_team.height:
        raise ValueError("tracking player club does not match either play side")
    window = window.select(
        pl.col("gameId").cast(pl.String).alias("game_id"),
        pl.col("playId").cast(pl.String).alias("play_id"),
        pl.col("nflId").cast(pl.String).alias("entity_id"),
        pl.when(pl.col("club") == pl.col("possessionTeam"))
        .then(pl.lit("offense"))
        .otherwise(pl.lit("defense"))
        .alias("side"),
        "t",
        pl.when(pl.col("playDirection") == "left")
        .then(120 - pl.col("x"))
        .otherwise(pl.col("x"))
        .alias("x"),
        pl.when(pl.col("playDirection") == "left")
        .then(WIDTH - pl.col("y"))
        .otherwise(pl.col("y"))
        .alias("y"),
    ).sort("game_id", "play_id", "entity_id", "t")
    entries = []
    for m in meta.to_dicts():
        reasons = []
        if m["direction_count"] != 1 or m["direction"] not in ["left", "right"]:
            reasons.append("inconsistent_pre_snap_direction")
        if m["ball_anchor_rows"] != 1 or m["ball_x"] is None:
            reasons.append("missing_unique_observed_ball_anchor")
        for flag in ["qbKneel", "qbSpike", "playNullifiedByPenalty"]:
            if m.get(flag) in [True, "Y", 1]:
                reasons.append("retrospective_" + flag)
        entries.append(
            {
                "game_id": str(m["gameId"]),
                "play_id": str(m["playId"]),
                "date": m["gameDate"],
                "season": m["season"],
                "week": m["week"],
                "offense": m["possessionTeam"],
                "defense": m["defensiveTeam"],
                "line_of_scrimmage": None
                if m["ball_x"] is None
                else (120 - m["ball_x"] if m["direction"] == "left" else m["ball_x"]),
                "dead_t": m["dead_t"],
                "synthetic": False,
                "source_exclusions": reasons,
            }
        )
    audit = {
        "file": raw.name,
        "rows": t.height,
        "schema": {c: str(d) for c, d in schema.items()},
        "plays": meta.height,
        "games": t["gameId"].n_unique(),
        "ball_rows": t["nflId"].null_count(),
        "inconsistent_snap_plays": snaps.height - valid.height,
        "missing_snap_plays": t.select("gameId", "playId").unique().height - snaps.height,
        "duplicate_keys": 0,
        "outside_field_rows": t.filter(
            ~pl.col("x").is_between(0, 120) | ~pl.col("y").is_between(0, WIDTH)
        ).height,
        "events": sorted(t["event"].drop_nulls().unique().to_list()),
        "frame_types": t.group_by("frameType").len().to_dicts(),
    }
    pre = (
        aligned.filter(pl.col("t").is_between(-3, -0.1))
        .sort("gameId", "playId", "nflId", "t")
        .with_columns(pl.col("t").diff().over("gameId", "playId", "nflId").alias("dt"))
    )
    audit["pre_snap_timestamp_deltas"] = pre.group_by("dt").len().sort("dt").to_dicts()
    return window, entries, audit


def outcome(rows: pl.DataFrame, sample: ForecastInput, dead_t: float | None) -> Reveal:
    valid = rows.filter(pl.col("x").is_between(0, 120) & pl.col("y").is_between(0, WIDTH))
    if dead_t is not None:
        valid = valid.filter(pl.col("t") <= dead_t + 1e-8)
    return reveal_from_rows(valid, sample).model_copy(
        update={
            "note": (
                "Observed historical 2022 endpoints; masks censor after ball dead or invalid "
                "coordinates. No inference about intent or quarterback perception."
            )
        }
    )


def build_dataset(raw: Path, out: Path, config: dict) -> Path:
    """Immutable canonical partitions, exclusion registry and numeric feature cache."""
    out.mkdir(parents=True, exist_ok=True)
    if (out / "dataset.json").exists():
        lineage = out / "canonical-lineage.json"
        if lineage.exists():
            recorded = json.loads(lineage.read_text())
            if recorded.get("policy_config_digest") != digest(config):
                raise ValueError("canonical policy differs; build a new version")
        summary = json.loads((out / "dataset.json").read_text())
        for name, expected in summary["feature_cache_hashes"].items():
            if file_hash(out / f"{name}.npy") != expected:
                raise ValueError(f"canonical cache checksum mismatch: {name}")
        return out
    games = pl.read_csv(raw / "games.csv", null_values=["NA"])
    plays = pl.read_csv(raw / "plays.csv", null_values=["NA"], infer_schema_length=20000)
    players = pl.read_csv(raw / "players.csv", null_values=["NA"])
    annotations = pl.read_csv(
        raw / "player_play.csv", null_values=["NA"], infer_schema_length=20000
    )
    unique_keys(games, ["gameId"], "games")
    unique_keys(plays, ["gameId", "playId"], "plays")
    unique_keys(players, ["nflId"], "players")
    unique_keys(annotations, ["gameId", "playId", "nflId"], "player_play")
    label_audit = {}
    for name, table in [("plays", plays), ("player_play", annotations)]:
        label_audit[name] = {
            c: {
                "rows": table.height,
                "missing": table[c].null_count(),
                "counts": table.group_by(c).len().sort("len", descending=True).head(30).to_dicts(),
                "availability": "annotation_target_only",
            }
            for c in table.columns
            if c
            in [
                "pff_passCoverage",
                "pff_manZone",
                "wasInitialPassRusher",
                "routeRan",
                "causedPressure",
            ]
        }
    write_json(out / "annotation-audit.json", label_audit)
    retrieval = json.loads((raw / "retrieval.json").read_text())
    dataset_id = (
        "nfl2022-"
        + digest([{k: r[k] for k in ["file", "bytes", "sha256"]} for r in retrieval["files"]])[:16]
    )
    metadata, exclusions, audits, partitions = [], [], [], []
    histories, anchors, lasts, velocities, ys, masks = [], [], [], [], [], []
    for week in range(1, 10):
        path = out / f"week-{week}.parquet"
        meta_path = out / f"week-{week}-metadata.json"
        if path.exists() and meta_path.exists():
            rows = pl.read_parquet(path)
            saved = json.loads(meta_path.read_text())
            entries, audit = saved["plays"], saved["audit"]
        else:
            rows, entries, audit = canonical_week(raw / f"tracking_week_{week}.csv", plays, games)
            rows.write_parquet(path, compression="zstd")
            write_json(meta_path, {"plays": entries, "audit": audit})
        audits.append(audit)
        partitions.append({"file": path.name, "sha256": file_hash(path), "rows": rows.height})
        groups = {
            key: g for key, g in rows.partition_by("game_id", "play_id", as_dict=True).items()
        }
        for meta in sorted(entries, key=lambda m: (m["game_id"], int(m["play_id"]))):
            key = (meta["game_id"], meta["play_id"])
            reasons = meta.pop("source_exclusions")
            if reasons:
                exclusions.append({"game_id": key[0], "play_id": key[1], "reasons": reasons})
                continue
            play = groups.get(key)
            if play is None:
                exclusions.append({"game_id": key[0], "play_id": key[1], "reasons": ["no_window"]})
                continue
            try:
                samples = [
                    input_from_rows(play, meta, c, 1.0, dataset_id) for c in config["cutoffs"]
                ]
                features = [build_features(s) for s in samples]
            except ValueError as error:
                exclusions.append({"game_id": key[0], "play_id": key[1], "reasons": [str(error)]})
                continue
            # Input eligibility is decided above, independently of future masks.
            f = features[0]
            order = np.r_[np.flatnonzero(~f.defenders), np.flatnonzero(f.defenders)]
            metadata.append(
                {**meta, "entity_ids": [f.ids[i] for i in order], "partition": path.name}
            )
            histories.append(np.array([x.history[order] for x in features], dtype=np.float32))
            anchors.append(f.anchor[order].astype(np.float32))
            lasts.append(np.array([x.last[order] for x in features], dtype=np.float32))
            velocities.append(np.array([x.velocity[order] for x in features], dtype=np.float32))
            player_targets, player_masks = [], []
            for h in config["horizons"]:
                reveal = outcome(play, samples[0].model_copy(update={"horizon": h}), meta["dead_t"])
                target = {r.entity_id: r.displacement for r in reveal.outcomes}
                player_targets.append([target[f.ids[i]] or (0, 0) for i in order[11:]])
                player_masks.append([target[f.ids[i]] is not None for i in order[11:]])
            ys.append(player_targets)
            masks.append(player_masks)
        print(
            f"week {week}: {audit['rows']} raw rows; "
            f"{len(metadata)} cumulative input-eligible plays",
            flush=True,
        )
    if not metadata:
        raise ValueError("no input-eligible plays")
    for name, value in [
        ("history", histories),
        ("anchor", anchors),
        ("last", lasts),
        ("velocity", velocities),
        ("target", ys),
        ("mask", masks),
    ]:
        np.save(out / f"{name}.npy", np.asarray(value))
    write_json(out / "plays.json", metadata)
    write_json(out / "exclusions.json", exclusions)
    write_json(
        out / "audit.json",
        {
            "weeks": audits,
            "annotations": label_audit,
            "excluded_plays": len(exclusions),
            "exclusion_reasons": dict(Counter(r for e in exclusions for r in e["reasons"])),
        },
    )
    summary = {
        "dataset_id": dataset_id,
        "synthetic": False,
        "season": 2022,
        "weeks": list(range(1, 10)),
        "games": len({m["game_id"] for m in metadata}),
        "plays": len(metadata),
        "rows": sum(a["rows"] for a in audits),
        "cadence_hz": 10,
        "partitions": partitions,
        "input_policy": "complete common input-eligible cohort; causal trailing histories",
        "target_policy": "exact endpoints; mask missing, out-of-field and after-ball-dead",
        "bengals_offense_plays": sum(m["offense"] == "CIN" for m in metadata),
        "bengals_defense_plays": sum(m["defense"] == "CIN" for m in metadata),
        "observed_defender_targets_by_horizon": np.array(masks).sum(axis=(0, 2)).tolist(),
        "source_files": retrieval,
        "annotation_input_columns": [],
        "local_replay_approved": True,
        "public_replay_approved": False,
        "m0_accepted": True,
        "feature_cache_hashes": {
            n: file_hash(out / f"{n}.npy")
            for n in ["history", "anchor", "last", "velocity", "target", "mask"]
        },
    }
    write_json(out / "dataset.json", summary)
    write_json(
        out / "canonical-lineage.json",
        {
            "adapter_version": "nfl-2022-causal-v1",
            "adapter_source_sha256": file_hash(Path(__file__)),
            "policy_config_digest": digest(config),
            "source_revision": config["source_revision"],
            "feature_version": "causal-geometry-v1",
            "target_version": "common-anchor-endpoint-v1",
        },
    )
    return out
