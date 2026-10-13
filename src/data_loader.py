"""Data loader module for ingesting, validating, and inspecting raw customer churn data."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.config import RAW_DATA_PATH, EXPECTED_COLUMNS, TARGET_COLUMN, ID_COLUMN


def validate_raw_data(df: pd.DataFrame, expected_columns: Optional[List[str]] = None) -> bool:
    """Validate that the input DataFrame contains all expected columns.

    Args:
        df: Input DataFrame to validate.
        expected_columns: List of column names expected in the dataset.

    Returns:
        True if all required columns are present.

    Raises:
        ValueError: If required columns are missing or if DataFrame is empty.
    """
    if df is None or df.empty:
        raise ValueError("Provided dataset is empty or None.")

    required = expected_columns or EXPECTED_COLUMNS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {missing}. "
            f"Expected {len(required)} columns."
        )
    return True


def load_raw_data(filepath: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco Customer Churn dataset from disk.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        pd.DataFrame: A fresh, independent copy of the raw dataset.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file fails schema validation.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found at '{path}'. "
            "Please ensure the dataset has been downloaded to data/raw/."
        )

    df = pd.read_csv(path)
    validate_raw_data(df)
    return df.copy()


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect dataset attributes including shape, duplicates, missing values, and target breakdown.

    Args:
        df: Input DataFrame.

    Returns:
        Dict[str, Any]: Summary metrics of the dataset.
    """
    whitespace_counts: Dict[str, int] = {}
    for col in df.columns:
        if df[col].dtype == object or df[col].dtype.name in ("string", "str"):
            spaces = int((df[col].astype(str).str.strip() == "").sum())
            if spaces > 0:
                whitespace_counts[col] = spaces

    summary = {
        "num_rows": int(df.shape[0]),
        "num_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(count) for col, count in df.isna().sum().items() if count > 0},
        "whitespace_counts": whitespace_counts,
        "duplicate_rows": int(df.duplicated().sum()),
    }

    if TARGET_COLUMN in df.columns:
        summary["target_distribution"] = df[TARGET_COLUMN].value_counts(dropna=False).to_dict()
        summary["target_proportions"] = (
            df[TARGET_COLUMN].value_counts(normalize=True, dropna=False) * 100
        ).round(2).to_dict()

    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("TELCO CUSTOMER CHURN: DATASET INSPECTION & VALIDATION")
    print("=" * 60)

    try:
        raw_df = load_raw_data()
        summary = inspect_dataset(raw_df)
        print(f"Dataset Path: {RAW_DATA_PATH}")
        print(f"Shape: {summary['num_rows']} rows x {summary['num_cols']} columns")
        print(f"Duplicate Rows: {summary['duplicate_rows']}")
        print(f"Missing Values (NaN): {summary['missing_values'] or 'None'}")
        print(f"Whitespace / Blank Fields: {summary['whitespace_counts'] or 'None'}")
        print(f"\nTarget '{TARGET_COLUMN}' Distribution:")
        for label, count in summary["target_distribution"].items():
            pct = summary["target_proportions"][label]
            print(f"  - {label}: {count} ({pct:.2f}%)")

        print("\nColumns and Data Types:")
        for col, dtype in summary["dtypes"].items():
            print(f"  - {col:<20} {dtype}")

        print("\nData validation successful! Dataset is ready for preprocessing pipeline.")
    except Exception as exc:
        print(f"\nData validation error: {exc}")

