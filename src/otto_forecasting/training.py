from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    gradient_clip: float,
    device: torch.device,
) -> list[dict[str, float | int]]:
    loss_function = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, patience // 3),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    remaining_patience = patience
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        training_loss = 0.0
        training_items = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = loss_function(predictions, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            training_loss += loss.item() * len(features)
            training_items += len(features)

        model.eval()
        validation_loss = 0.0
        validation_items = 0
        with torch.inference_mode():
            for features, targets in validation_loader:
                features = features.to(device)
                targets = targets.to(device)
                loss = loss_function(model(features), targets)
                validation_loss += loss.item() * len(features)
                validation_items += len(features)

        train_average = training_loss / training_items
        validation_average = validation_loss / validation_items
        scheduler.step(validation_average)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_average,
                "validation_loss": validation_average,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        if validation_average < best_loss - 1e-6:
            best_loss = validation_average
            best_state = copy.deepcopy(model.state_dict())
            remaining_patience = patience
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break

    model.load_state_dict(best_state)
    return history


def predict_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for features, targets in loader:
            output = model(features.to(device)).cpu().numpy()
            predictions.append(output)
            actuals.append(targets.numpy())
    return np.concatenate(predictions), np.concatenate(actuals)


def save_history(history: list[dict[str, float | int]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(history, indent=2), encoding="utf-8")
