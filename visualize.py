#!/usr/bin/env python3
"""
Generate comprehensive visualizations for the hybrid nanofluid ML project.

Outputs are written to the `reports/` directory and cover:
* Training vs test feature distributions
* Feature correlation matrix
* Predicted vs actual plots for classical and neural models
* Residual diagnostics for both models
* Neural-network training history
* Cross-validation metrics summary
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

FEATURES = ["M", "S", "K", "phi1", "phi2", "Ec", "Pr", "eta"]
TARGETS = ["f3", "f5"]
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean = pd.read_csv("clean_dataset.csv")
    train = pd.read_csv("data/processed/train_dataset.csv")
    test = pd.read_csv("data/processed/test_dataset.csv")
    return clean, train, test


def load_models():
    classical = joblib.load("models/classical_baseline.pkl")
    metadata_path = Path("models/neural_network_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    artifact = torch.load(
        "models/neural_network.pt",
        map_location="cpu",
        weights_only=False,
    )
    state_dict = artifact["state_dict"]
    feature_mean = np.array(artifact["feature_mean"], dtype=float)
    feature_std = np.array(artifact["feature_std"], dtype=float)

    net = nn.Sequential(
        nn.Linear(8, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )
    net.load_state_dict(state_dict)
    net.eval()

    def neural_predict(X: np.ndarray) -> np.ndarray:
        scaled = (X - feature_mean) / (feature_std + 1e-8)
        with torch.no_grad():
            return net(torch.tensor(scaled, dtype=torch.float32)).numpy()

    return classical, neural_predict, metadata


def plot_feature_distributions(train: pd.DataFrame, test: pd.DataFrame) -> None:
    cols = ["M", "S", "K", "phi1", "phi2", "Ec", "Pr", "eta"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    bins = 20
    for ax, col in zip(axes.ravel(), cols):
        ax.hist(train[col], bins=bins, alpha=0.6, label="Train", color="#1f77b4")
        ax.hist(test[col], bins=bins, alpha=0.6, label="Test", color="#ff7f0e")
        ax.set_title(col)
        ax.set_ylabel("Count")
    axes[0, 0].legend()
    fig.suptitle("Feature Distributions: Train vs Test", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(REPORTS_DIR / "feature_distributions.png", dpi=200)
    plt.close(fig)


def plot_correlation_matrix(clean: pd.DataFrame) -> None:
    corr = clean[FEATURES + TARGETS].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Feature/Target Correlation Matrix")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "correlation_matrix.png", dpi=200)
    plt.close(fig)


def scatter_and_residual_plots(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    for idx, target in enumerate(TARGETS):
        # Parity plot
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_true[:, idx], y_pred[:, idx], alpha=0.7, edgecolor="k")
        diag = np.linspace(y_true[:, idx].min(), y_true[:, idx].max(), 100)
        ax.plot(diag, diag, "r--", linewidth=1)
        ax.set_xlabel(f"Actual {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(f"{name}: Actual vs Predicted ({target})")
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / f"{name.lower()}_parity_{target}.png", dpi=200)
        plt.close(fig)

        # Residual plot
        residuals = y_true[:, idx] - y_pred[:, idx]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_pred[:, idx], residuals, alpha=0.7, edgecolor="k")
        ax.axhline(0, color="r", linestyle="--")
        ax.set_xlabel(f"Predicted {target}")
        ax.set_ylabel("Residual (actual - pred)")
        ax.set_title(f"{name}: Residuals vs Predicted ({target})")
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / f"{name.lower()}_residuals_{target}.png", dpi=200)
        plt.close(fig)


def plot_neural_training(history: dict | list | None) -> None:
    if not history:
        return
    records = history if isinstance(history, list) else history.get("history", [])
    if not records:
        return
    epochs = [entry["epoch"] for entry in records]
    train_loss = [entry["train_loss"] for entry in records]
    val_loss = [entry["val_loss"] for entry in records]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_loss, label="Train")
    ax.plot(epochs, val_loss, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Neural Network Training History")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "neural_training_history.png", dpi=200)
    plt.close(fig)


def plot_cv_metrics(summary_path: Path) -> None:
    if not summary_path.exists():
        return
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    cv_metrics = data.get("metrics", {}).get("cross_validation")
    if not cv_metrics:
        return
    names = ["MAE", "RMSE", "R2_f3", "R2_f5"]
    means = [cv_metrics[name]["mean"] for name in names]
    stds = [cv_metrics[name]["std"] for name in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, means, yerr=stds, capsize=6, color="#1f77b4")
    ax.set_title("5-fold Cross-validation Metrics (XGBoost)")
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "cross_validation_summary.png", dpi=200)
    plt.close(fig)


def main() -> None:
    clean_df, train_df, test_df = load_data()
    classical_model, neural_predict, nn_metadata = load_models()

    plot_feature_distributions(train_df, test_df)
    plot_correlation_matrix(clean_df)

    X_train = train_df[FEATURES].to_numpy(dtype=float)
    y_train = train_df[TARGETS].to_numpy(dtype=float)
    X_test = test_df[FEATURES].to_numpy(dtype=float)
    y_test = test_df[TARGETS].to_numpy(dtype=float)

    y_pred_classical = classical_model.predict(X_test)
    scatter_and_residual_plots("Classical", y_test, y_pred_classical)

    y_pred_neural = neural_predict(X_test)
    scatter_and_residual_plots("Neural", y_test, y_pred_neural)

    plot_neural_training(nn_metadata.get("history"))
    plot_cv_metrics(Path("reports/verification_summary.json"))


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8")
    main()
