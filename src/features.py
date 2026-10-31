import numpy as np
import pandas as pd

from src.data_utils import normalize_name


TARGET_NAME_PARTS = (
    "label",
    "target",
    "launder",
    "suspicious",
    "sar",
    "typology",
    "alert",
)

IDENTIFIER_NAME_PARTS = (
    "id",
    "account",
    "customer",
    "counterparty",
    "address",
    "phone",
    "email",
)


def parse_binary_label(
    data: pd.DataFrame,
    label_column: str,
    positive_label: str,
) -> pd.Series:
    """Convert an explicitly specified positive label into binary values."""
    if label_column not in data.columns:
        raise ValueError(
            f"Label column {label_column!r} not found. "
            f"Available columns: {data.columns.tolist()}"
        )

    labels = data[label_column].astype("string").str.strip()

    if labels.isna().any():
        raise ValueError(
            "The label column has missing values. Review them before training."
        )

    positive_mask = labels.str.casefold() == str(positive_label).casefold()

    if not positive_mask.any():
        observed = sorted(labels.unique().tolist())
        raise ValueError(
            f"Positive label {positive_label!r} was not found. "
            f"Observed labels: {observed[:20]}"
        )

    result = positive_mask.astype(int)

    if result.nunique() != 2:
        raise ValueError("The label column must contain two classes.")

    return result


def build_features(
    data: pd.DataFrame,
    label_column: str,
    amount_column: str | None,
    timestamp_column: str | None,
    sender_column: str | None,
    receiver_column: str | None,
    type_column: str | None,
    extra_id_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build numeric and categorical features.

    Sender and receiver counts are computed over the supplied file. For
    time-sensitive evaluation, calculate these using only information
    available at the prediction time.
    """
    extra_id_columns = extra_id_columns or []
    features = pd.DataFrame(index=data.index)

    excluded = {label_column, *extra_id_columns}

    if amount_column and amount_column in data.columns:
        amount = pd.to_numeric(
            data[amount_column]
            .astype("string")
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False),
            errors="coerce",
        )
        features["amount"] = amount
        features["log_amount"] = np.log1p(amount.clip(lower=0))
        features["amount_is_negative"] = (amount < 0).astype(int)
        excluded.add(amount_column)

    if timestamp_column and timestamp_column in data.columns:
        timestamp = pd.to_datetime(
            data[timestamp_column],
            errors="coerce",
            utc=True,
        )
        features["transaction_hour"] = timestamp.dt.hour
        features["transaction_day_of_week"] = timestamp.dt.dayofweek
        features["transaction_month"] = timestamp.dt.month
        features["is_weekend"] = (
            timestamp.dt.dayofweek >= 5
        ).astype(float)
        excluded.add(timestamp_column)

    if sender_column and sender_column in data.columns:
        features["sender_count_in_file"] = (
            data[sender_column].map(data[sender_column].value_counts())
        )
        excluded.add(sender_column)

    if receiver_column and receiver_column in data.columns:
        features["receiver_count_in_file"] = (
            data[receiver_column].map(data[receiver_column].value_counts())
        )
        excluded.add(receiver_column)

    if (
        sender_column and sender_column in data.columns
        and receiver_column and receiver_column in data.columns
    ):
        features["sender_equals_receiver"] = (
            data[sender_column].astype("string")
            == data[receiver_column].astype("string")
        ).astype(int)

    if type_column and type_column in data.columns:
        features[type_column] = data[type_column].astype("string")
        excluded.add(type_column)

    # Include other numerical columns and low-cardinality categorical columns,
    # but avoid label-like fields and likely identifiers.
    for column in data.columns:
        if column in excluded or column in features.columns:
            continue

        normalized = normalize_name(column)

        if any(part in normalized for part in TARGET_NAME_PARTS):
            continue

        if any(part in normalized for part in IDENTIFIER_NAME_PARTS):
            continue

        if pd.api.types.is_numeric_dtype(data[column]):
            features[column] = pd.to_numeric(data[column], errors="coerce")
        elif pd.api.types.is_bool_dtype(data[column]):
            features[column] = data[column].astype(float)
        elif data[column].nunique(dropna=True) <= 20:
            features[column] = data[column].astype("string")

    features = features.dropna(axis=1, how="all")

    if features.empty:
        raise ValueError(
            "No usable features found. Check the dataset columns "
            "and feature settings."
        )

    return features
