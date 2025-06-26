from typing import Any, Dict
import numpy as np
import pytorch_lightning as pl  # noqa
import torch
import torch.nn.functional as F
from torch import nn


class MLP(pl.LightningModule):
    def __init__(
        self, input_size: int, x_col: str = "emb", y_col: str = "avg_rating"
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.x_col = x_col
        self.y_col = y_col
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x = batch[self.x_col]
        y = batch[self.y_col].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x = batch[self.x_col]
        y = batch[self.y_col].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def normalize(x: np.ndarray, axis: int = -1, order: int = 2) -> np.ndarray:
    y = np.atleast_1d(np.linalg.norm(x, order, axis))
    y[y == 0.0] = 1.0
    return x / np.expand_dims(y, axis)
