"""One feature path for training, evaluation and API inference."""

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data

from disguise_lab.contracts import LENGTH, WIDTH, ForecastInput


def rotate(points: np.ndarray, left: bool, vector: bool = False) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if not left:
        return points.copy()
    return -points if vector else np.array([LENGTH, WIDTH]) - points


def wrap_angle(degrees: float) -> float:
    return degrees % 360


@dataclass(frozen=True)
class Features:
    ids: tuple[str, ...]
    history: np.ndarray  # nodes, time, [relative x/y, trailing vx/vy, side, time]
    anchor: np.ndarray
    last: np.ndarray
    velocity: np.ndarray
    defenders: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    context: np.ndarray

    def graph(self) -> Data:
        return Data(
            x=torch.tensor(self.history, dtype=torch.float32),
            edge_index=torch.tensor(self.edge_index, dtype=torch.long),
            edge_attr=torch.tensor(self.edge_attr, dtype=torch.float32),
            defender=torch.tensor(self.defenders),
            anchor=torch.tensor(self.anchor, dtype=torch.float32),
            last=torch.tensor(self.last, dtype=torch.float32),
            velocity=torch.tensor(self.velocity, dtype=torch.float32),
            context=torch.tensor(self.context, dtype=torch.float32),
            num_nodes=len(self.ids),
        )


def build_features(sample: ForecastInput) -> Features:
    histories, anchors, lasts, velocities, sides = [], [], [], [], []
    for entity in sample.entities:
        times = np.array([p.t for p in entity.observations])
        xy = np.array([[p.x, p.y] for p in entity.observations])
        vel = np.zeros_like(xy)
        vel[1:] = np.diff(xy, axis=0) / np.diff(times)[:, None]
        # Fixed causal 21-point grid, trailing carry-forward only. Never backfill.
        grid = np.round(np.linspace(sample.cutoff - 2, sample.cutoff, 21), 8)
        indices = np.searchsorted(times, grid + 1e-8, side="right") - 1
        if np.any(indices < 0):
            raise ValueError("insufficient two-second history; no future backfill")
        side = float(entity.side == "defense")
        relative = xy[indices] - [sample.line_of_scrimmage, WIDTH / 2]
        histories.append(
            np.column_stack([relative / 10, vel[indices] / 5, np.full(21, side), grid / 3])
        )
        anchors.append([entity.anchor.x, entity.anchor.y])
        lasts.append(xy[-1])
        velocities.append(vel[-1])
        sides.append(side)
    last = np.array(lasts, dtype=np.float32)
    velocity = np.array(velocities, dtype=np.float32)
    sides_array = np.array(sides)
    send, receive = np.where(~np.eye(len(last), dtype=bool))
    delta = last[send] - last[receive]
    attrs = np.column_stack(
        [
            delta / 10,
            np.linalg.norm(delta, axis=1) / 10,
            (velocity[send] - velocity[receive]) / 5,
            sides_array[send] == sides_array[receive],
        ]
    )
    return Features(
        tuple(e.entity_id for e in sample.entities),
        np.array(histories, dtype=np.float32),
        np.array(anchors, dtype=np.float32),
        last,
        velocity,
        sides_array.astype(bool),
        np.array([send, receive]),
        attrs,
        np.tile([sample.cutoff, sample.horizon], (len(last), 1)),
    )


def geometry_matrix(features: Features) -> np.ndarray:
    own = features.history[:, -1]
    context = []
    for side in (False, True):
        subset = own[features.defenders == side]
        context.extend([subset.mean(axis=0), subset.std(axis=0)])
    pooled = np.tile(np.concatenate(context), (len(own), 1))
    delta = features.last[:, None] - features.last[None, :]
    distance = np.linalg.norm(delta, axis=-1)
    opposite = features.defenders[:, None] != features.defenders[None, :]
    nearest = np.min(np.where(opposite, distance, np.inf), axis=1)
    return np.column_stack([own, pooled, nearest / 10])
