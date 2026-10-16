"""Immutable columnar partitions and analytical queries for audited local data."""

from pathlib import Path

import duckdb
import polars as pl

from disguise_lab.audit import file_hash, unique_keys
from disguise_lab.contracts import digest

CANONICAL_SCHEMA = pl.Schema(
    {
        "game_id": pl.String,
        "play_id": pl.String,
        "entity_id": pl.String,
        "side": pl.String,
        "t": pl.Float64,
        "x": pl.Float64,
        "y": pl.Float64,
    }
)


def write_partition(frame: pl.DataFrame, path: Path) -> dict:
    if set(frame.columns) != set(CANONICAL_SCHEMA):
        raise ValueError("canonical table columns must match typed tracking contract")
    typed = frame.cast(CANONICAL_SCHEMA, strict=True)
    unique_keys(typed, ["game_id", "play_id", "entity_id", "t"], "canonical tracking")
    if typed.null_count().sum_horizontal()[0]:
        raise ValueError("canonical rows require explicit valid values")
    invalid = typed.filter(
        ~pl.all_horizontal(pl.col("t", "x", "y").is_finite())
        | ~pl.col("side").is_in(["offense", "defense"])
        | ~pl.col("x").is_between(0, 120)
        | ~pl.col("y").is_between(0, 160 / 3)
    )
    if invalid.height:
        raise ValueError("canonical rows require finite, in-field coordinates and valid sides")
    if path.exists():
        raise FileExistsError("canonical partitions are immutable; use a new version")
    path.parent.mkdir(parents=True, exist_ok=True)
    typed.sort(["game_id", "play_id", "entity_id", "t"]).write_parquet(path)
    return {
        "path": str(path),
        "rows": typed.height,
        "sha256": file_hash(path),
        "schema_hash": digest({c: str(t) for c, t in CANONICAL_SCHEMA.items()}),
    }


def summarize_parquet(path: Path) -> dict:
    with duckdb.connect() as db:
        # Parameter binding avoids SQL interpolation of user paths.
        row = db.execute(
            "SELECT COUNT(*), COUNT(DISTINCT game_id), "
            "COUNT(DISTINCT (game_id,play_id)), MIN(t), MAX(t) "
            "FROM read_parquet(?)",
            [str(path)],
        ).fetchone()
    if row is None:
        raise RuntimeError("columnar summary returned no result")
    return dict(zip(["rows", "games", "plays", "min_time", "max_time"], row, strict=True))


def opponent_split(
    metadata: list[dict], test_opponents: set[str], validation_opponents: set[str]
) -> dict:
    if not test_opponents or not validation_opponents or test_opponents & validation_opponents:
        raise ValueError("distinct predeclared validation/test opponent sets required")
    groups: dict[str, set[str]] = {}
    for p in metadata:
        groups.setdefault(p["game_id"], set()).update([p["offense"], p["defense"]])
    mapping = {}
    for game, teams in groups.items():
        if teams & test_opponents:
            mapping[game] = "test"
        elif teams & validation_opponents:
            mapping[game] = "validation"
        else:
            mapping[game] = "train"
    if set(mapping.values()) != {"train", "validation", "test"}:
        raise ValueError("opponent protocol has an empty partition")
    value = {
        "protocol": "entire-games-with-selected-opponents",
        "games": mapping,
        "test_opponents": sorted(test_opponents),
        "validation_opponents": sorted(validation_opponents),
        "priority": "test games exclude validation overlap; report overlap explicitly",
    }
    return {**value, "split_id": digest(value)}
