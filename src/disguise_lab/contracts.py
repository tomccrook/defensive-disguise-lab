"""Input-only service contracts: no outcome annotation fields exist here."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LENGTH = 120.0
WIDTH = 160 / 3
FEATURE_VERSION: Literal["causal-geometry-v1"] = "causal-geometry-v1"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Observation(Strict):
    t: float
    x: float = Field(ge=0, le=LENGTH)
    y: float = Field(ge=0, le=WIDTH)


class EntityHistory(Strict):
    entity_id: str
    side: Literal["offense", "defense"]
    observations: tuple[Observation, ...]
    anchor: Observation


class ForecastInput(Strict):
    dataset_id: str
    game_id: str
    play_id: str
    cutoff: float
    horizon: float
    anchor_seconds: float = -1.0
    line_of_scrimmage: float = Field(ge=0, le=LENGTH)
    entities: tuple[EntityHistory, ...]
    feature_version: Literal["causal-geometry-v1"] = FEATURE_VERSION

    @model_validator(mode="after")
    def causal_contract(self) -> ForecastInput:
        if self.cutoff not in (-1.0, -0.5, -0.1) or self.horizon not in (0.5, 1.0):
            raise ValueError("cutoff or horizon outside frozen target contract")
        if self.anchor_seconds != -1.0:
            raise ValueError("common anchor must be -1.0 seconds")
        if len(self.entities) != 22 or len({e.entity_id for e in self.entities}) != 22:
            raise ValueError("complete unique 22-player input required; ball excluded in v1")
        if sum(e.side == "defense" for e in self.entities) != 11:
            raise ValueError("expected 11 defenders")
        for entity in self.entities:
            obs = entity.observations
            times = [p.t for p in obs]
            if len(obs) < 2 or times != sorted(set(times)):
                raise ValueError("history must be ordered, unique and have at least two frames")
            if times[-1] > self.cutoff + 1e-8 or times[0] < self.cutoff - 2 - 1e-8:
                raise ValueError("history outside permitted causal window")
            if times[0] > self.cutoff - 2 + 1e-8:
                raise ValueError("complete two-second history required; no future backfill")
            if self.cutoff - times[-1] > 0.100001:
                raise ValueError("last observation is stale")
            if any(b - a > 0.150001 for a, b in zip(times, times[1:], strict=False)):
                raise ValueError("input gap exceeds 0.15 seconds")
            if abs(entity.anchor.t - self.anchor_seconds) > 1e-6:
                raise ValueError("exact common anchor required")
            if entity.anchor.t > self.cutoff:
                raise ValueError("anchor not observed at cutoff")
            matching = [p for p in obs if abs(p.t - entity.anchor.t) < 1e-6]
            if not matching or matching[0] != entity.anchor:
                raise ValueError("anchor must match permitted history")
            for a, b in zip(obs, obs[1:], strict=False):
                if ((b.x - a.x) ** 2 + (b.y - a.y) ** 2) ** 0.5 / (b.t - a.t) > 15:
                    raise ValueError("implausible tracking jump")
        return self

    def input_hash(self) -> str:
        # Canonical entity ordering makes cache/hash independent of storage order.
        value = self.model_dump(mode="json")
        value["entities"] = sorted(value["entities"], key=lambda e: e["entity_id"])
        return digest(value)


class DefenderMode(Strict):
    weight: float = Field(ge=0, le=1)
    mean_displacement: tuple[float, float]
    covariance: tuple[tuple[float, float], tuple[float, float]]

    @model_validator(mode="after")
    def positive_covariance(self):
        validate_covariance(self.covariance)
        return self


class DefenderForecast(Strict):
    entity_id: str
    anchor: tuple[float, float]
    mean_displacement: tuple[float, float]
    covariance: tuple[tuple[float, float], tuple[float, float]]
    modes: tuple[DefenderMode, ...] = ()

    @model_validator(mode="after")
    def valid_distribution(self):
        validate_covariance(self.covariance)
        if self.modes and abs(sum(m.weight for m in self.modes) - 1) > 1e-5:
            raise ValueError("mixture weights must sum to one")
        return self


def validate_covariance(covariance):
    (a, b), (c, d) = covariance
    if abs(b - c) > 1e-5 or a <= 0 or a * d - b * c <= 0:
        raise ValueError("symmetric positive definite covariance required")


class Forecast(Strict):
    status: Literal["ok"] = "ok"
    input_hash: str
    model_id: str
    dataset_id: str
    calibration_id: str
    feature_version: str
    target_version: str = "common-anchor-endpoint-v1"
    split_id: str
    cutoff: float
    anchor_seconds: float
    horizon: float
    units: Literal["yards"] = "yards"
    distribution: Literal["marginal_bivariate_gaussian", "shared_component_mixture"] = (
        "marginal_bivariate_gaussian"
    )
    region_note: str = "Nominal 80% marginal region; inspect measured coverage. Not joint."
    defenders: tuple[DefenderForecast, ...]


class Outcome(Strict):
    entity_id: str
    displacement: tuple[float, float] | None


class Reveal(Strict):
    game_id: str
    play_id: str
    horizon: float
    outcomes: tuple[Outcome, ...]
    future: dict[str, tuple[Observation, ...]]
    note: str = "Observed fixture future; no claim about intent or quarterback perception."
