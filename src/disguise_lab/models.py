"""Matched distribution heads across temporal, pooled and edge models."""

from typing import Literal

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

from disguise_lab.features import Features, geometry_matrix

Family = Literal["persistence", "velocity", "geometry", "temporal", "pooled", "graph"]


class Baseline:
    def __init__(self, family: Family, static_geometry: bool = False, iterations: int = 80):
        self.family = family
        self.static_geometry = static_geometry
        self.anchor_residual = False
        self.estimator = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=iterations,
                max_leaf_nodes=31 if iterations > 80 else 15,
                l2_regularization=1,
                random_state=1729,
                early_stopping=False,
            )
        )
        self.covariance = np.eye(2)
        self.context_covariance: dict[tuple[int, int], np.ndarray] = {}

    def matrix(self, f: Features) -> np.ndarray:
        if getattr(self, "static_geometry", False):
            from dataclasses import replace

            history = f.history.copy()
            history[:, :, 2:4] = 0
            f = replace(f, history=history)
        return np.column_stack([geometry_matrix(f), f.context])[f.defenders]

    def points(self, f: Features) -> np.ndarray:
        if self.family == "geometry":
            point = self.estimator.predict(self.matrix(f))
            if getattr(self, "anchor_residual", False):
                point += (f.last - f.anchor)[f.defenders]
            return point
        point = f.last - f.anchor
        if self.family == "velocity":
            point += f.velocity * (f.context[:, 1] - f.context[:, 0])[:, None]
        return point[f.defenders]

    def fit(self, examples: list[tuple[Features, np.ndarray, np.ndarray]]) -> None:
        if self.family == "geometry":
            x = np.concatenate([self.matrix(f)[mask] for f, _, mask in examples])
            y = np.concatenate([target[mask] for _, target, mask in examples])
            self.estimator.fit(x, y)
        residual = np.concatenate(
            [target[mask] - self.points(f)[mask] for f, target, mask in examples]
        )
        self.covariance = np.cov(residual.T) + np.eye(2) * 0.15**2

    def predict(self, f: Features) -> tuple[np.ndarray, np.ndarray]:
        mean = self.points(f)
        c = [-1.0, -0.5, -0.1].index(float(f.context[0, 0]))
        h = [0.5, 1.0].index(float(f.context[0, 1]))
        covariance = getattr(self, "context_covariance", {}).get((c, h), self.covariance)
        return mean, np.tile(covariance, (len(mean), 1, 1))


class EdgeLayer(MessagePassing):
    def __init__(self, width: int):
        super().__init__(aggr="mean")
        self.message_mlp = nn.Sequential(
            nn.Linear(width * 2 + 6, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.update_mlp = nn.Sequential(nn.Linear(width * 2, width), nn.SiLU())

    def forward(self, x, edge_index, edge_attr):
        message = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x + self.update_mlp(torch.cat([x, message], dim=-1))

    def message(self, x_i, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class MovementModel(nn.Module):
    def __init__(
        self, family: Family, width: int = 24, static_history: bool = False, no_side: bool = False
    ):
        super().__init__()
        self.family = family
        self.static_history = static_history
        self.no_side = no_side
        self.encoder = nn.GRU(6, width, batch_first=True)
        self.layers = (
            nn.ModuleList([EdgeLayer(width), EdgeLayer(width)]) if family == "graph" else None
        )
        head_width = width * (3 if family == "pooled" else 1) + 2
        self.head = nn.Sequential(nn.Linear(head_width, width), nn.SiLU(), nn.Linear(width, 5))

    def encode(self, data):
        x = data.x
        if getattr(self, "static_history", False):
            x = x[:, -1:].clone()
            x[:, :, 2:4] = 0
        if getattr(self, "no_side", False):
            x = x.clone()
            x[:, :, 4] = 0
        _, state = self.encoder(x)
        encoded = state[-1]
        if self.layers is not None:
            for layer in self.layers:
                edge_attr = data.edge_attr
                if getattr(self, "no_side", False):
                    edge_attr = edge_attr.clone()
                    edge_attr[:, -1] = 0
                encoded = layer(encoded, data.edge_index, edge_attr)
        if self.family == "pooled":
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(len(encoded), dtype=torch.long, device=encoded.device)
            groups = batch * 2 + data.defender.long()
            context = scatter(
                encoded, groups, dim=0, dim_size=(int(batch.max()) + 1) * 2, reduce="mean"
            )
            encoded = torch.cat([encoded, context[batch * 2], context[batch * 2 + 1]], dim=-1)
        return encoded

    def forward(self, data):
        raw = self.head(torch.cat([self.encode(data), data.context], dim=-1))[data.defender]
        mean = raw[:, :2] + (data.last - data.anchor)[data.defender]
        chol = torch.zeros(len(raw), 2, 2, device=raw.device, dtype=raw.dtype)
        chol[:, 0, 0] = F.softplus(raw[:, 2]) + 0.15
        chol[:, 1, 0] = raw[:, 3]
        chol[:, 1, 1] = F.softplus(raw[:, 4]) + 0.15
        return mean, chol @ chol.transpose(-1, -2)


class SharedMixtureModel(MovementModel):
    """One shared scene component; conditional defenders remain independent."""

    def __init__(self, family: Family, width: int = 24, components: int = 3):
        super().__init__(family, width)
        self.components = components
        head_width = width * (3 if family == "pooled" else 1) + 2
        self.head = nn.Sequential(
            nn.Linear(head_width, width), nn.SiLU(), nn.Linear(width, components * 5)
        )
        self.weight_head = nn.Sequential(
            nn.Linear(head_width, width), nn.SiLU(), nn.Linear(width, components)
        )

    def forward(self, data):
        encoded = self.encode(data)
        augmented = torch.cat([encoded, data.context], dim=-1)
        raw = self.head(augmented)[data.defender].reshape(-1, self.components, 5)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(len(encoded), dtype=torch.long, device=encoded.device)
        pooled = scatter(augmented, batch, dim=0, reduce="mean")
        log_weights = self.weight_head(pooled).log_softmax(-1)
        mean = raw[:, :, :2] + (data.last - data.anchor)[data.defender, None, :]
        chol = torch.zeros(*raw.shape[:2], 2, 2, device=raw.device, dtype=raw.dtype)
        chol[:, :, 0, 0] = F.softplus(raw[:, :, 2]) + 0.15
        chol[:, :, 1, 0] = raw[:, :, 3]
        chol[:, :, 1, 1] = F.softplus(raw[:, :, 4]) + 0.15
        return mean, chol @ chol.transpose(-1, -2), log_weights


def mixture_loss(mean, covariance, log_weights, target, mask, batch):
    logp = torch.distributions.MultivariateNormal(mean, covariance).log_prob(target[:, None, :])
    observed = torch.where(mask[:, None], logp, torch.zeros_like(logp))
    joint = scatter(observed, batch, dim=0, reduce="sum")
    counts = scatter(mask.to(mean.dtype), batch, dim=0, reduce="sum")
    valid = counts > 0
    if not valid.any():
        raise ValueError("batch has no observed targets")
    return (-torch.logsumexp(log_weights[valid] + joint[valid], dim=-1) / counts[valid]).mean()


def masked_loss(mean, covariance, target, mask):
    if not mask.any():
        raise ValueError("batch has no observed targets")
    loss = -torch.distributions.MultivariateNormal(mean[mask], covariance[mask]).log_prob(
        target[mask]
    )
    return loss.mean()


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def matched_width(graph_width: int = 24) -> int:
    target = parameter_count(MovementModel("graph", graph_width))
    return min(
        range(8, 97), key=lambda w: abs(parameter_count(MovementModel("pooled", w)) - target)
    )
