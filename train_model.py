#!/usr/bin/env python3
"""
Training entrypoints for the hybrid nanofluid ML project.

Although day-to-day experimentation happens in the notebooks, this script
captures the same logic so training can be reproduced in a headless
environment (CI or batch jobs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from torch import nn
from torch.utils.data import DataLoader, Dataset

import preprocess

FEATURES = ["M", "S", "K", "phi1", "phi2", "Ec", "Pr", "eta"]
TARGETS = ["f3", "f5"]


def ensure_processed_data(train_path: Path, test_path: Path) -> None:
    """Run preprocessing if processed splits are missing."""
    if train_path.exists() and test_path.exists():
        return
    preprocess.main([])


def load_splits(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def _compute_metrics(y_true: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for idx, target in enumerate(TARGETS):
        metrics[f"{target}_MAE"] = mean_absolute_error(y_true[:, idx], preds[:, idx])
        metrics[f"{target}_RMSE"] = mean_squared_error(
            y_true[:, idx], preds[:, idx]
        ) ** 0.5
        metrics[f"{target}_R2"] = r2_score(y_true[:, idx], preds[:, idx])
    metrics["joint_MAE"] = mean_absolute_error(y_true, preds)
    metrics["joint_RMSE"] = np.sqrt(np.mean((y_true - preds) ** 2))
    return metrics


def train_classical_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    estimator: str,
    output_path: Path,
) -> Dict[str, float]:
    X_train = train_df[FEATURES].to_numpy(dtype=float)
    y_train = train_df[TARGETS].to_numpy(dtype=float)
    X_test = test_df[FEATURES].to_numpy(dtype=float)
    y_test = test_df[TARGETS].to_numpy(dtype=float)

    def make_pipeline(model):
        return Pipeline([("scale", StandardScaler()), ("model", model)])

    estimators = {
        "random_forest": RandomForestRegressor(
            n_estimators=400, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": MultiOutputRegressor(
            GradientBoostingRegressor(random_state=42)
        ),
        "knn": MultiOutputRegressor(KNeighborsRegressor(n_neighbors=5)),
        "xgboost": MultiOutputRegressor(
            XGBRegressor(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
                verbosity=0,
            )
        ),
        "lightgbm": MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=700,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=0.5,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        ),
    }
    if estimator not in estimators:
        raise ValueError(f"Unknown estimator '{estimator}'")

    pipeline = make_pipeline(estimators[estimator])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    metrics = _compute_metrics(y_test, preds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    return metrics


class GradientDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        self.X = torch.tensor(inputs, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def build_network() -> nn.Module:
    return nn.Sequential(
        nn.Linear(8, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )


def train_neural_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: Path,
    metadata_path: Path,
    epochs: int = 500,
    patience: int = 40,
) -> Dict[str, float]:
    X_full = train_df[FEATURES].to_numpy(dtype=float)
    y_full = train_df[TARGETS].to_numpy(dtype=float)
    X_test = test_df[FEATURES].to_numpy(dtype=float)
    y_test = test_df[TARGETS].to_numpy(dtype=float)

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, shuffle=True
    )
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    def scale(features: np.ndarray) -> np.ndarray:
        return (features - mean) / std

    loaders = {
        "train": DataLoader(
            GradientDataset(scale(X_train), y_train), batch_size=32, shuffle=True
        ),
        "val": DataLoader(
            GradientDataset(scale(X_val), y_val), batch_size=32, shuffle=False
        ),
    }
    test_dataset = GradientDataset(scale(X_test), y_test)

    model = build_network()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    patience_counter = 0

    def run_epoch(loader: DataLoader, train: bool) -> float:
        total_loss = 0.0
        total_samples = 0
        if train:
            model.train()
        else:
            model.eval()
        with torch.set_grad_enabled(train):
            for xb, yb in loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * xb.size(0)
                total_samples += xb.size(0)
        return total_loss / max(1, total_samples)

    for _ in range(epochs):
        train_loss = run_epoch(loaders["train"], train=True)
        val_loss = run_epoch(loaders["val"], train=False)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    def predict(dataset: Dataset) -> np.ndarray:
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        preds = []
        model.eval()
        with torch.no_grad():
            for xb, _ in loader:
                preds.append(model(xb).numpy())
        return np.vstack(preds)

    preds = predict(test_dataset)
    metrics = _compute_metrics(y_test, preds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "feature_mean": mean, "feature_std": std}, output_path)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_mean": mean.tolist(),
                "feature_std": std.tolist(),
                "metrics": metrics,
            },
            f,
            indent=2,
        )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML models for f3/f5 prediction.")
    parser.add_argument(
        "--model-type",
        choices=["classical", "neural"],
        default="classical",
        help="Select which model family to train.",
    )
    parser.add_argument(
        "--estimator",
        choices=["random_forest", "gradient_boosting", "knn", "xgboost", "lightgbm"],
        default="random_forest",
        help="Classical estimator to use (ignored for neural training).",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/processed/train_dataset.csv"),
        help="Path to the processed training CSV.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("data/processed/test_dataset.csv"),
        help="Path to the processed test CSV.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("models"),
        help="Directory where trained artifacts will be stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_processed_data(args.train_path, args.test_path)
    train_df, test_df = load_splits(args.train_path, args.test_path)

    if args.model_type == "classical":
        output_path = args.artifact_dir / "classical_baseline.pkl"
        metrics = train_classical_model(train_df, test_df, args.estimator, output_path)
    else:
        output_path = args.artifact_dir / "neural_network.pt"
        metadata_path = args.artifact_dir / "neural_network_metadata.json"
        metrics = train_neural_model(train_df, test_df, output_path, metadata_path)

    print(json.dumps({"model_type": args.model_type, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
