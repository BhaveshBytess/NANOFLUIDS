#!/usr/bin/env python3
"""
Data preprocessing pipeline for the hybrid nanofluid project.

The logic in this module follows the agent specification and the First
Principles Framework (FPF):
* preserve the solver outputs (f3, f5) while removing numerical blow-ups,
* keep the original governing features and their physical ordering, and
* enforce unit consistency (e.g., convert rotation/angle S to radians).
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Sequence
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RAW_INPUT_COLUMNS = ["M", "S", "K", "phi1", "phi2", "Ec", "Pr", "eta"]
OUTPUT_COLUMNS = ["f3", "f5"]
MAX_GRADIENT_MAG = 1e4
ETA_RANGE = (0.0, 5.0)


def _column_letters_to_index(col: str) -> int:
    """Convert column letters (e.g., 'AA') to a zero-based index."""
    value = 0
    for char in col:
        if not char.isalpha():
            continue
        value = value * 26 + (ord(char.upper()) - ord("A") + 1)
    return value - 1


def _read_shared_strings(zf: ZipFile) -> List[str]:
    """Extract shared strings from the Excel archive."""
    shared_strings: List[str] = []
    name = "xl/sharedStrings.xml"
    if name not in zf.namelist():
        return shared_strings

    root = ET.fromstring(zf.read(name))
    for si in root.findall("main:si", NS):
        parts: List[str] = []
        for node in si.findall(".//main:t", NS):
            parts.append(node.text or "")
        shared_strings.append("".join(parts))
    return shared_strings


def load_raw_dataset(path: Path) -> pd.DataFrame:
    """
    Load the Excel dataset without relying on optional dependencies.

    We parse the OOXML payload manually so the agent can operate even when
    `openpyxl` is unavailable in the execution environment.
    """
    with ZipFile(path) as zf:
        shared_strings = _read_shared_strings(zf)
        worksheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        sheet_data = worksheet.find("main:sheetData", NS)
        if sheet_data is None:
            raise ValueError("No sheetData node found in worksheet")

        rows: List[List[str | None]] = []
        for row in sheet_data.findall("main:row", NS):
            values: List[str | None] = []
            for cell in row.findall("main:c", NS):
                ref = cell.get("r", "")
                letters = "".join(ch for ch in ref if ch.isalpha())
                idx = _column_letters_to_index(letters)
                while len(values) <= idx:
                    values.append(None)

                cell_value: str | None = None
                raw_value = cell.find("main:v", NS)
                if raw_value is not None:
                    text = raw_value.text or ""
                    if cell.get("t") == "s" and text.isdigit():
                        cell_value = shared_strings[int(text)]
                    else:
                        cell_value = text

                values[idx] = cell_value
            rows.append(values)

    if not rows:
        raise ValueError("Empty worksheet")

    header = rows[0]
    data_rows = rows[1:]
    df = pd.DataFrame(data_rows, columns=header)
    df = df.dropna(axis=1, how="all")  # drop empty trailing column
    df = df.rename(columns={"f3(eta)": "f3", "f5(eta)": "f5"})
    return df


def _coerce_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _convert_s_to_radians(df: pd.DataFrame) -> pd.DataFrame:
    """Convert rotation/angle S from degrees whenever the value exceeds 3.5."""
    mask = df["S"] > 3.5
    if mask.any():
        df.loc[mask, "S"] = np.deg2rad(df.loc[mask, "S"])
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the mandatory cleaning rules from AGENT.md.

    Steps:
    1. Ensure numeric dtypes for all physical quantities.
    2. Drop rows with missing f3/f5 values.
    3. Remove numerical blow-ups (|f3| or |f5| > 1e4).
    4. Convert S to radians for all apparent degree entries (S > 3.5).
    5. Filter eta to the physically meaningful [0, 5] interval.
    6. Deduplicate and restore canonical column ordering.
    """
    df = df.copy()
    df = _coerce_numeric(df, RAW_INPUT_COLUMNS + OUTPUT_COLUMNS)
    df = df.dropna(subset=OUTPUT_COLUMNS)
    stability_mask = (df["f3"].abs() <= MAX_GRADIENT_MAG) & (
        df["f5"].abs() <= MAX_GRADIENT_MAG
    )
    df = df.loc[stability_mask]
    df = _convert_s_to_radians(df)
    eta_low, eta_high = ETA_RANGE
    eta_mask = df["eta"].between(eta_low, eta_high)
    df = df.loc[eta_mask]
    df = df.dropna(subset=RAW_INPUT_COLUMNS + OUTPUT_COLUMNS)
    df = df.drop_duplicates()

    ordered_columns = RAW_INPUT_COLUMNS + OUTPUT_COLUMNS
    return df[ordered_columns].reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the cleaned data into train/test partitions."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_datasets(
    clean_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    clean_path: Path,
    train_path: Path,
    test_path: Path,
) -> None:
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    clean_df.to_csv(clean_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the hybrid nanofluid dataset per agent specs."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("survey_sample_data.xlsx"),
        help="Raw Excel dataset path.",
    )
    parser.add_argument(
        "--clean-path",
        type=Path,
        default=Path("clean_dataset.csv"),
        help="Output path for the cleaned dataset.",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/processed/train_dataset.csv"),
        help="Output path for the training split.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("data/processed/test_dataset.csv"),
        help="Output path for the test split.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples to reserve for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the data split.",
    )
    return parser.parse_args(args=args)


def main(cli_args: Sequence[str] | None = None) -> None:
    args = parse_args(cli_args)
    df = load_raw_dataset(args.input_path)
    clean_df = clean_dataset(df)
    train_df, test_df = split_dataset(
        clean_df, test_size=args.test_size, random_state=args.random_state
    )
    save_datasets(
        clean_df=clean_df,
        train_df=train_df,
        test_df=test_df,
        clean_path=args.clean_path,
        train_path=args.train_path,
        test_path=args.test_path,
    )


if __name__ == "__main__":
    main()
