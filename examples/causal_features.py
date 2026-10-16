"""Build one forecast input from independent synthetic player histories."""

import polars as pl

from disguise_lab.features import build_features
from disguise_lab.fixtures import generate, input_from_rows


def main() -> None:
    rows, metadata = generate(6, 1)
    play = metadata[0]
    selected = rows.filter(pl.col("game_id") == play["game_id"])
    sample = input_from_rows(selected, play, -0.1, 1)
    features = build_features(sample)
    print(f"Cutoff: {sample.cutoff}s; target: +{sample.horizon}s")
    print(f"History array (players, time, features): {features.history.shape}")
    print(f"Directed edges: {features.edge_index.shape[1]}")
    print(f"Input identity: {sample.input_hash()}")
    assert all(p.t <= sample.cutoff for entity in sample.entities for p in entity.observations)


if __name__ == "__main__":
    main()
