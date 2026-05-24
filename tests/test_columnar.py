import polars as pl
import pytest

from disguise_lab.columnar import opponent_split, summarize_parquet, write_partition
from disguise_lab.fixtures import generate


def test_columnar_identity_immutability_and_queries(tmp_path):
    frame, _ = generate(6, 1)
    path = tmp_path / "fixture.parquet"
    manifest = write_partition(frame, path)
    assert manifest["rows"] == frame.height
    assert summarize_parquet(path)["games"] == 6
    with pytest.raises(FileExistsError):
        write_partition(frame, path)
    with pytest.raises(ValueError):
        write_partition(pl.concat([frame, frame.head(1)]), tmp_path / "duplicate.parquet")
    with pytest.raises(ValueError):
        write_partition(
            frame.with_columns(pl.lit(float("nan")).alias("x")), tmp_path / "nan.parquet"
        )
    with pytest.raises(ValueError):
        write_partition(
            frame.with_columns(pl.lit("unknown").alias("side")), tmp_path / "side.parquet"
        )


def test_opponent_holdout_keeps_entire_games():
    rows = [
        {"game_id": "a", "offense": "CIN", "defense": "BAL"},
        {"game_id": "a", "offense": "BAL", "defense": "CIN"},
        {"game_id": "b", "offense": "KC", "defense": "BUF"},
        {"game_id": "c", "offense": "SEA", "defense": "LAR"},
    ]
    split = opponent_split(rows, {"BAL"}, {"BUF"})
    assert split["games"] == {"a": "test", "b": "validation", "c": "train"}
