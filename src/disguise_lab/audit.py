"""Read-only local download audit. No credentials, automatic rules acceptance or mirrors."""

import hashlib
from pathlib import Path
from typing import Any, cast

import polars as pl


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unique_keys(table: pl.DataFrame, keys: list[str], name: str) -> None:
    if table.select(keys).is_duplicated().any():
        raise ValueError(f"{name} duplicate keys: {keys}")


def audit_directory(root: Path) -> dict:
    paths = sorted(root.glob("*.csv"))
    if not paths:
        return {"status": "missing_files", "usable_for_training": False}
    files = [
        {
            "name": p.name,
            "bytes": p.stat().st_size,
            "sha256": file_hash(p),
            "schema": {
                k: str(v) for k, v in pl.scan_csv(p, null_values=["NA"]).collect_schema().items()
            },
        }
        for p in paths
    ]
    game_path, play_path = root / "games.csv", root / "plays.csv"
    if not game_path.exists() or not play_path.exists():
        return {
            "status": "missing_game_play_metadata",
            "files": files,
            "usable_for_training": False,
        }
    games = pl.read_csv(game_path, null_values=["NA"])
    plays = pl.read_csv(play_path, null_values=["NA"], infer_schema_length=10000)
    unique_keys(games, ["gameId"], "games")
    unique_keys(plays, ["gameId", "playId"], "plays")
    tracking_paths = [p for p in paths if p.name.startswith("tracking_")]
    summaries: list[dict[str, Any]] = []
    all_ids: set[int] = set()
    bengals_off, bengals_def = 0, 0
    for path in tracking_paths:
        schema = pl.scan_csv(path, null_values=["NA"]).collect_schema()
        required = {"gameId", "playId", "nflId", "time", "x", "y", "event"}
        if not required.issubset(schema):
            summaries.append(
                {
                    "file": path.name,
                    "status": "incompatible_schema",
                    "missing": sorted(required - set(schema)),
                }
            )
            continue
        frame = "frameId" if "frameId" in schema else "frame.id"
        # Projection and per-file streaming. Canonicalisation later partitions by complete play.
        t = (
            pl.scan_csv(path, null_values=["NA"])
            .select([*sorted(required), frame])
            .collect(engine="streaming")
        )
        t = t.with_columns(
            pl.col("time").str.to_datetime(time_zone="UTC").alias("timestamp"),
            pl.col("nflId").cast(pl.String).fill_null("ball").alias("entity"),
        )
        keys = ["gameId", "playId", "entity", frame]
        duplicates = int(t.select(keys).is_duplicated().sum())
        unique_keys(t, keys, "tracking")
        present = t.select(["gameId", "playId"]).unique()
        joins = present.join(plays, on=["gameId", "playId"], how="left", validate="m:1")
        if joins["possessionTeam"].null_count():
            raise ValueError("tracking play missing play metadata")
        joins = joins.join(
            games.select(["gameId", "homeTeamAbbr", "visitorTeamAbbr"]),
            on="gameId",
            how="left",
            validate="m:1",
        )
        if joins["homeTeamAbbr"].null_count():
            raise ValueError("tracking game missing game metadata")
        offense = joins["possessionTeam"]
        defense = joins.with_columns(
            pl.when(pl.col("possessionTeam") == pl.col("homeTeamAbbr"))
            .then(pl.col("visitorTeamAbbr"))
            .otherwise(pl.col("homeTeamAbbr"))
            .alias("defense")
        )["defense"]
        bengals_off += int((offense == "CIN").sum())
        bengals_def += int((defense == "CIN").sum())
        ids = t["gameId"].unique().to_list()
        all_ids.update(ids)
        clocks = t.sort(["gameId", "playId", "entity", "timestamp"]).with_columns(
            pl.col("timestamp")
            .diff()
            .over(["gameId", "playId", "entity"])
            .dt.total_milliseconds()
            .alias("dt_ms")
        )
        cadence = clocks.filter(pl.col("dt_ms") > 0)["dt_ms"]
        snaps = (
            t.filter(pl.col("event") == "ball_snap")
            .group_by(["gameId", "playId"])
            .agg(
                pl.col("timestamp").n_unique().alias("snap_times"),
                pl.col(frame).n_unique().alias("snap_frames"),
                pl.col("timestamp").first().alias("snap"),
            )
        )
        valid_snaps = snaps.filter((pl.col("snap_times") == 1) & (pl.col("snap_frames") == 1))
        aligned = clocks.join(
            valid_snaps.select(["gameId", "playId", "snap"]),
            on=["gameId", "playId"],
            how="inner",
            validate="m:1",
        ).with_columns(
            ((pl.col("timestamp") - pl.col("snap")).dt.total_milliseconds() / 1000).alias(
                "relative_t"
            )
        )
        coverage = []
        for cutoff in [-1.0, -0.5, -0.1]:
            hist = (
                aligned.filter(
                    (pl.col("entity") != "ball")
                    & (pl.col("relative_t") >= cutoff - 2 - 1e-8)
                    & (pl.col("relative_t") <= cutoff + 1e-8)
                )
                .group_by(["gameId", "playId", "entity"])
                .agg(
                    pl.col("relative_t").min().alias("start"),
                    pl.col("relative_t").max().alias("end"),
                    pl.col("dt_ms").max().alias("max_gap"),
                )
            )
            eligible = (
                hist.filter(
                    (pl.col("start") <= cutoff - 2 + 1e-6)
                    & (pl.col("end") >= cutoff - 0.100001)
                    & (pl.col("max_gap") <= 150.001)
                )
                .group_by(["gameId", "playId"])
                .len()
            )
            coverage.append(
                {
                    "cutoff": cutoff,
                    "complete_22player_histories": eligible.filter(pl.col("len") == 22).height,
                }
            )
        targets = {
            str(h): aligned.filter(
                (pl.col("entity") != "ball") & ((pl.col("relative_t") - h).abs() < 1e-6)
            ).height
            for h in [0.5, 1.0]
        }
        bounds = int(
            t.filter(~pl.col("x").is_between(0, 120) | ~pl.col("y").is_between(0, 160 / 3)).height
        )
        summaries.append(
            {
                "file": path.name,
                "status": "audited",
                "rows": t.height,
                "games": ids,
                "plays": present.height,
                "duplicates": duplicates,
                "out_of_field_rows": bounds,
                "ball_rows": int((t["entity"] == "ball").sum()),
                "nonzero_timestamp_delta_ms_median": float(cast(float, cadence.median()))
                if len(cadence)
                else None,
                "nonzero_timestamp_delta_ms_values": sorted(cadence.unique().to_list()),
                "duplicate_entity_timestamp_rows": int(
                    t.select(["gameId", "playId", "entity", "timestamp"]).is_duplicated().sum()
                ),
                "clock_note": (
                    "Timestamp resolution may differ from frame cadence; "
                    "duplicates require source clock audit."
                ),
                "missing_snap_plays": present.height - snaps.height,
                "inconsistent_snap_plays": snaps.height - valid_snaps.height,
                "history_coverage": coverage,
                "candidate_target_rows_not_unique_verified_targets": targets,
                "direction_available": "playDirection" in schema,
                "events": sorted(t["event"].drop_nulls().unique().to_list()),
            }
        )
    selected_games = games.filter(pl.col("gameId").is_in(sorted(all_ids)))
    annotation = {}
    for path in [play_path, root / "player_play.csv"]:
        if path.exists():
            schema = pl.scan_csv(path, null_values=["NA"]).collect_schema()
            columns = [
                c
                for c in schema
                if c.startswith("pff_")
                or c in ["wasInitialPassRusher", "routeRan", "causedPressure"]
            ]
            if columns:
                labels = pl.scan_csv(path, null_values=["NA"]).select(columns).collect()
                annotation[path.name] = {
                    c: {
                        "missing": labels[c].null_count(),
                        "rows": len(labels),
                        "unique": labels[c].n_unique(),
                        "availability": "target_only_not_input",
                    }
                    for c in columns
                }
    return {
        "status": "audited_not_approved",
        "source": str(root),
        "files": files,
        "tracking": summaries,
        "tracking_games": len(all_ids),
        "season": selected_games["season"].unique().to_list(),
        "game_dates": selected_games.select(
            ["gameId", "gameDate", "homeTeamAbbr", "visitorTeamAbbr"]
        ).to_dicts(),
        "bengals_offensive_plays": bengals_off,
        "bengals_defending_plays": bengals_def,
        "annotations": annotation,
        "usable_for_training": False,
        "gate_note": (
            "Audit is not rights approval. Freeze sufficient multi-game/Bengals cohort first."
        ),
    }
