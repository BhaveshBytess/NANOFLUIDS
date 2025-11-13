#!/usr/bin/env python3
"""
Inference utilities for predicting (f3, f5) from hybrid nanofluid parameters.

Supports both the saved classical pipeline (scikit-learn) and the PyTorch
neural network defined in the notebooks / train_model.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

FEATURES = ["M", "S", "K", "phi1", "phi2", "Ec", "Pr", "eta"]
DEFAULT_CLASSICAL_PATH = Path("models/classical_baseline.pkl")
DEFAULT_NEURAL_PATH = Path("models/neural_network.pt")
DEFAULT_NEURAL_METADATA = Path("models/neural_network_metadata.json")


def _ensure_s_radians(arr: np.ndarray) -> np.ndarray:
    """Convert S from degrees to radians when values exceed 3.5."""
    arr = arr.copy()
    mask = arr[:, 1] > 3.5
    arr[mask, 1] = np.deg2rad(arr[mask, 1])
    return arr


def _load_inputs(values: Sequence[float] | None, csv_path: Path | None) -> np.ndarray:
    if values is not None and csv_path is not None:
        raise ValueError("Provide either inline --values or --input-csv, not both.")
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        missing = [col for col in FEATURES if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")
        data = df[FEATURES].to_numpy(dtype=float)
        return _ensure_s_radians(data)
    if values is not None:
        if len(values) != len(FEATURES):
            raise ValueError(f"--values expects {len(FEATURES)} numbers.")
        data = np.array(values, dtype=float).reshape(1, -1)
        return _ensure_s_radians(data)
    raise ValueError("Either --values or --input-csv must be provided.")


def _build_network() -> nn.Module:
    return nn.Sequential(
        nn.Linear(8, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )


def predict_classical(inputs: np.ndarray, model_path: Path | str = DEFAULT_CLASSICAL_PATH) -> np.ndarray:
    pipeline = joblib.load(model_path)
    return pipeline.predict(inputs)


def predict_neural(
    inputs: np.ndarray,
    model_path: Path | str = DEFAULT_NEURAL_PATH,
    metadata_path: Path | str = DEFAULT_NEURAL_METADATA,
) -> np.ndarray:
    model_data = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(model_data, dict) and "state_dict" in model_data:
        state_dict = model_data["state_dict"]
        feature_mean = np.array(model_data.get("feature_mean"), dtype=float)
        feature_std = np.array(model_data.get("feature_std"), dtype=float)
    else:
        state_dict = model_data
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        feature_mean = np.array(metadata["feature_mean"], dtype=float)
        feature_std = np.array(metadata["feature_std"], dtype=float)

    if feature_mean.shape[0] != inputs.shape[1]:
        raise ValueError("Feature normalization vector does not match inputs.")

    inputs_scaled = (inputs - feature_mean) / (feature_std + 1e-8)
    model = _build_network()
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        tensor_inputs = torch.tensor(inputs_scaled, dtype=torch.float32)
        preds = model(tensor_inputs).numpy()
    return preds


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict f3/f5 from hybrid nanofluid inputs.")
    parser.add_argument(
        "--model-type",
        choices=["classical", "neural"],
        default="classical",
        help="Select which trained artifact to use.",
    )
    parser.add_argument(
        "--values",
        type=float,
        nargs=len(FEATURES),
        metavar=tuple(FEATURES),
        help="Inline feature values in the canonical order.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="CSV file with the required feature columns.",
    )
    parser.add_argument(
        "--classical-path",
        type=Path,
        default=DEFAULT_CLASSICAL_PATH,
        help="Path to the saved classical pipeline.",
    )
    parser.add_argument(
        "--neural-path",
        type=Path,
        default=DEFAULT_NEURAL_PATH,
        help="Path to the saved neural-network weights (.pt).",
    )
    parser.add_argument(
        "--neural-metadata",
        type=Path,
        default=DEFAULT_NEURAL_METADATA,
        help="JSON metadata for neural normalization (fallback when .pt lacks stats).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    inputs = _load_inputs(args.values, args.input_csv)
    if args.model_type == "classical":
        preds = predict_classical(inputs, args.classical_path)
    else:
        preds = predict_neural(inputs, args.neural_path, args.neural_metadata)
    df = pd.DataFrame(preds, columns=["f3", "f5"])
    print(df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
