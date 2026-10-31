import re
from pathlib import Path

import pandas as pd


def normalize_name(value: str) -> str:
    """Normalize a column name for case- and punctuation-insensitive matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def load_data(path: Path) -> pd.DataFrame:
    """Load a CSV and perform basic validation."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place the CSV at that path or update config.py."
        )

    data = pd.read_csv(path)
    data.columns = [str(column).strip() for column in data.columns]

    if data.empty:
        raise ValueError("The input CSV contains no rows.")

    return data


def inspect_data(data: pd.DataFrame) -> None:
    """Print the CSV schema, missing values, and example records."""
    print(f"Rows: {len(data):,}")
    print(f"Columns: {len(data.columns):,}")

    print("\nColumns and data types:")
    for column in data.columns:
        print(f"  {column}: {data[column].dtype}")

    print("\nMissing values:")
    missing = data.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0].to_string() or "None")

    print("\nPossible label columns:")
    possible_labels = [
        column for column in data.columns
        if any(word in normalize_name(column) for word in
               ("label", "target", "launder", "suspicious", "sar"))
    ]
    print(", ".join(possible_labels) if possible_labels else "None detected")

    print("\nFirst five rows:")
    print(data.head().to_string(index=False))
