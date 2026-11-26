"""Red de regresión por cuantiles y entrenamiento en CPU."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


QUANTILES = (0.1, 0.5, 0.9)
HIDDEN_LAYERS = (32, 16)
BATCH_SIZE = 64
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0001
MAX_EPOCHS = 200
PATIENCE = 20


class DurationMLP(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        first, second = HIDDEN_LAYERS
        self.layers = nn.Sequential(
            nn.Linear(n_features, first), nn.ReLU(),
            nn.Linear(first, second), nn.ReLU(),
            nn.Linear(second, len(QUANTILES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        lower, median_gap, upper_gap = F.softplus(self.layers(features)).unbind(dim=-1)
        median = lower + median_gap
        return torch.stack((lower, median, median + upper_gap), dim=-1)


def pinball_loss(
    estimates: torch.Tensor, actual: torch.Tensor, quantiles: torch.Tensor
) -> torch.Tensor:
    error = actual[:, None] - estimates
    return torch.maximum(quantiles * error, (quantiles - 1) * error).mean()


def fit_mlp(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, seed: int
) -> tuple[DurationMLP, dict]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = DurationMLP(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    batches = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train / 60)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_features = torch.from_numpy(x_val)
    val_minutes = torch.from_numpy(y_val / 60)
    quantiles = torch.tensor(QUANTILES, dtype=torch.float32)
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for features, actual_minutes in batches:
            optimizer.zero_grad()
            loss = pinball_loss(model(features), actual_minutes, quantiles)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(features)

        model.eval()
        with torch.inference_mode():
            val_loss = pinball_loss(model(val_features), val_minutes, quantiles).item()
        history.append({
            "epoch": epoch,
            "train_pinball_minutes": loss_sum / len(x_train),
            "validation_pinball_minutes": val_loss,
        })
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        if epoch - best_epoch >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("La red no produjo una pérdida de validación finita.")
    model.load_state_dict(best_state)
    return model, {
        "hidden_layers": HIDDEN_LAYERS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "history": history,
    }


@torch.inference_mode()
def predict_seconds(model: DurationMLP, features: np.ndarray) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(features)).numpy() * 60
