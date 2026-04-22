from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from disguise_lab.features import build_features
from disguise_lab.fixtures import generate, input_from_rows
from disguise_lab.nfl import canonical_week, outcome


def source_fixture(tmp_path):
    rows, metadata = generate(6, 1)
    meta = metadata[0]
    rows = rows.filter(pl.col("game_id") == meta["game_id"])
    raw = []
    epoch = datetime(2022, 9, 11, 17)
    for r in rows.to_dicts():
        raw.append(
            {
                "gameId": 1,
                "playId": 2,
                "nflId": int(r["entity_id"][1:]) + (100 if r["side"] == "defense" else 0),
                "frameId": int(round((r["t"] + 3) * 10)) + 1,
                "frameType": "SNAP"
                if r["t"] == 0
                else "BEFORE_SNAP"
                if r["t"] < 0
                else "AFTER_SNAP",
                "time": str(epoch + timedelta(seconds=r["t"])),
                "club": "CIN" if r["side"] == "offense" else "PIT",
                "playDirection": "right",
                "x": r["x"],
                "y": r["y"],
                "event": "ball_snap" if r["t"] == 0 else "tackle" if r["t"] == 0.8 else None,
            }
        )
    for t in sorted(rows["t"].unique().to_list()):
        raw.append(
            {
                **raw[0],
                "nflId": None,
                "frameId": int(round((t + 3) * 10)) + 1,
                "time": str(epoch + timedelta(seconds=t)),
                "x": meta["line_of_scrimmage"],
                "event": "ball_snap" if t == 0 else None,
                "frameType": "SNAP" if t == 0 else "BEFORE_SNAP" if t < 0 else "AFTER_SNAP",
            }
        )
    path = tmp_path / "tracking.csv"
    pl.DataFrame(raw).write_csv(path)
    plays = pl.DataFrame(
        {
            "gameId": [1],
            "playId": [2],
            "possessionTeam": ["CIN"],
            "defensiveTeam": ["PIT"],
            "qbKneel": [False],
            "qbSpike": [False],
            "playNullifiedByPenalty": [False],
        }
    )
    games = pl.DataFrame({"gameId": [1], "season": [2022], "week": [1], "gameDate": ["9/11/2022"]})
    return path, plays, games


def test_real_adapter_causal_boundary_and_censoring(tmp_path):
    path, plays, games = source_fixture(tmp_path)
    rows, metadata, audit = canonical_week(path, plays, games)
    assert audit["missing_snap_plays"] == 0
    assert audit["duplicate_keys"] == 0
    meta = metadata[0]
    sample = input_from_rows(rows, meta, -0.1, 1, "fixture")
    f = build_features(sample)
    assert not any(
        x.displacement is not None for x in outcome(rows, sample, meta["dead_t"]).outcomes
    )
    assert all(
        x.displacement is not None
        for x in outcome(rows, sample.model_copy(update={"horizon": 0.5}), meta["dead_t"]).outcomes
    )
    raw = pl.read_csv(path).with_columns(
        pl.when(pl.col("frameType") != "BEFORE_SNAP").then(5).otherwise(pl.col("x")).alias("x")
    )
    raw.write_csv(path)
    changed, _, _ = canonical_week(path, plays, games)
    second = input_from_rows(changed, meta, -0.1, 1, "fixture")
    assert sample.input_hash() == second.input_hash()
    np.testing.assert_array_equal(f.history, build_features(second).history)


def test_duplicate_join_and_bad_identity_rejected(tmp_path):
    path, plays, games = source_fixture(tmp_path)
    raw = pl.read_csv(path)
    pl.concat([raw, raw.head(1)]).write_csv(path)
    with pytest.raises(ValueError, match="duplicate keys"):
        canonical_week(path, plays, games)
    raw.with_columns(
        pl.when(pl.col("nflId") == 1).then(pl.lit("WRONG")).otherwise(pl.col("club")).alias("club")
    ).write_csv(path)
    with pytest.raises(ValueError, match="club"):
        canonical_week(path, plays, games)
