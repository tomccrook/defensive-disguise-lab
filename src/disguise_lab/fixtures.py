"""Synthetic generator, explicitly unrelated to actual NFL game records."""

from datetime import date, timedelta

import numpy as np
import polars as pl

from disguise_lab.contracts import (
    EntityHistory,
    ForecastInput,
    Observation,
    Outcome,
    Reveal,
    digest,
)

DATASET = "synthetic-1729-v1"
TEAMS = ["CIN", "BAL", "PIT", "CLE", "BUF", "KC", "LAR", "SEA"]


def generate(games: int = 18, plays: int = 8, seed: int = 1729) -> tuple[pl.DataFrame, list]:
    rng = np.random.default_rng(seed)
    rows, metadata = [], []
    times = np.round(np.arange(-3, 1.1001, 0.1), 6)
    for game in range(games):
        game_id = f"S{game:03d}"
        offense = "CIN" if game % 3 == 0 else TEAMS[game % len(TEAMS)]
        defense = TEAMS[(game + 1) % len(TEAMS)]
        if defense == offense:
            defense = "CLE"
        for play in range(plays):
            play_id = f"P{play:03d}"
            los = 45.0 + rng.uniform(-8, 8)
            base_off = np.column_stack(
                [
                    los + np.array([-1] * 5 + [-4, -6, -1, -1, -1, -1]),
                    [22, 24, 26.7, 29, 31, 26.7, 25, 8, 15, 39, 46],
                ]
            )
            base_def = np.column_stack(
                [
                    los + np.array([1] * 4 + [4] * 3 + [7] * 2 + [12] * 2),
                    [21, 25, 29, 33, 18, 27, 36, 8, 45, 20, 34],
                ]
            )
            bases = np.vstack([base_off, base_def]) + rng.normal(0, 0.25, (22, 2))
            latent = rng.choice([-1, 1])
            motion = rng.normal(0, 0.08, (22, 2))
            motion[8, 1] = latent * rng.uniform(0.8, 1.6)
            response = rng.normal(0, 0.12, (22, 2))
            response[11:, 1] += latent * 0.4
            future_vel = rng.normal(0, 0.55, (22, 2))
            future_vel[11:, 0] += [-3.5, -3.2, -3.8, -3.5, 1.5, 1.8, 1.4, 2.8, 2.8, 3.2, 3.2]
            future_vel[11:, 1] += latent * np.linspace(-1.2, 1.2, 11)
            # Deliberately shifted late fixtures, not an NFL shift claim.
            if game >= 15:
                future_vel[11:, 1] += 1.0
            for entity in range(22):
                for t in times:
                    pre_t = min(t, 0)
                    xy = bases[entity] + motion[entity] * (pre_t + 1)
                    xy += response[entity] * max(pre_t + 0.8, 0)
                    if t > 0:
                        xy += future_vel[entity] * t
                    rows.append(
                        {
                            "game_id": game_id,
                            "play_id": play_id,
                            "entity_id": f"{'D' if entity >= 11 else 'O'}{entity % 11 + 1:02d}",
                            "side": "defense" if entity >= 11 else "offense",
                            "t": float(t),
                            "x": float(xy[0]),
                            "y": float(xy[1]),
                        }
                    )
            metadata.append(
                {
                    "game_id": game_id,
                    "play_id": play_id,
                    "date": str(date(2000, 1, 1) + timedelta(days=game)),
                    "season": None,
                    "offense": offense,
                    "defense": defense,
                    "line_of_scrimmage": los,
                    "synthetic": True,
                }
            )
    return pl.DataFrame(rows), metadata


def input_from_rows(
    rows: pl.DataFrame, meta: dict, cutoff: float, horizon: float, dataset_id: str = DATASET
) -> ForecastInput:
    # Filter first. Source columns/annotations never pass through to inference.
    allowed = rows.filter((pl.col("t") <= cutoff + 1e-8) & (pl.col("t") >= cutoff - 2 - 1e-8))
    entities = []
    for key, group in allowed.sort(["entity_id", "t"]).group_by("entity_id", maintain_order=True):
        obs = tuple(Observation(t=r["t"], x=r["x"], y=r["y"]) for r in group.to_dicts())
        anchors = [p for p in obs if abs(p.t + 1) < 1e-6]
        if not anchors:
            raise ValueError("missing common anchor")
        entities.append(
            EntityHistory(
                entity_id=key[0], side=group["side"][0], observations=obs, anchor=anchors[0]
            )
        )
    return ForecastInput(
        dataset_id=dataset_id,
        game_id=meta["game_id"],
        play_id=meta["play_id"],
        cutoff=cutoff,
        horizon=horizon,
        line_of_scrimmage=meta["line_of_scrimmage"],
        entities=tuple(entities),
    )


def reveal_from_rows(rows: pl.DataFrame, sample: ForecastInput) -> Reveal:
    outcomes, future = [], {}
    for entity in sample.entities:
        if entity.side != "defense":
            continue
        entity_rows = rows.filter(pl.col("entity_id") == entity.entity_id).sort("t")
        target = entity_rows.filter((pl.col("t") - sample.horizon).abs() < 1e-6)
        displacement = (
            None
            if target.is_empty()
            else (float(target["x"][0] - entity.anchor.x), float(target["y"][0] - entity.anchor.y))
        )
        outcomes.append(Outcome(entity_id=entity.entity_id, displacement=displacement))
        future[entity.entity_id] = tuple(
            Observation(t=r["t"], x=r["x"], y=r["y"])
            for r in entity_rows.filter(
                (pl.col("t") > sample.cutoff) & (pl.col("t") <= sample.horizon)
            ).to_dicts()
        )
    return Reveal(
        game_id=sample.game_id,
        play_id=sample.play_id,
        horizon=sample.horizon,
        outcomes=tuple(outcomes),
        future=future,
    )


def chronological_split(metadata: list[dict]) -> dict:
    games = sorted({(m["date"], m["game_id"]) for m in metadata})
    if len(games) < 6:
        raise ValueError("at least six independent games required for fixture split")
    n_train, n_val = int(len(games) * 2 / 3), max(1, len(games) // 6)
    mapping = {
        g: "train" if i < n_train else "validation" if i < n_train + n_val else "test"
        for i, (_, g) in enumerate(games)
    }
    value = {
        "protocol": "chronological-whole-games",
        "games": mapping,
        "synthetic": True,
        "final_nfl_test_exposed": False,
    }
    return {**value, "split_id": digest(value)}
