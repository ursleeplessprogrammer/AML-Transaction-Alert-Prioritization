import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_model(features: pd.DataFrame) -> Pipeline:
    """Create preprocessing and logistic regression as one pipeline."""
    numeric_columns = features.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        column for column in features.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def make_review_signals(
    data: pd.DataFrame,
    amount_column: str | None,
) -> pd.Series:
    """
    Create simple, visible review notes.

    These are not model explanations or findings of wrongdoing.
    """
    signals = pd.Series(
        "Model score exceeded review threshold",
        index=data.index,
        dtype="string",
    )

    if amount_column and amount_column in data.columns:
        amount = pd.to_numeric(
            data[amount_column]
            .astype("string")
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False),
            errors="coerce",
        )

        if amount.notna().any():
            cutoff = amount.quantile(0.99)
            high_amount = amount >= cutoff
            signals.loc[high_amount] = (
                "Model score exceeded threshold; "
                "amount is at or above dataset 99th percentile"
            )

    return signals
