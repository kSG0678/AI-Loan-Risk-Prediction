"""Reusable, leakage-safe preprocessing for the loan approval dataset.

Run ``python ml/src/data_preprocessing.py`` from any working directory to
prepare and validate a reproducible train/test split without training a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __package__:
    from .feature_engineering import LoanFeatureEngineer, engineer_features
else:
    from feature_engineering import LoanFeatureEngineer, engineer_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "ml" / "data" / "raw" / "train_u6lujuX_CVtuZ9i.csv"
)
TARGET_COLUMN = "Loan_Status"
NON_PREDICTIVE_COLUMNS = ("Loan_ID",)
TEST_SIZE = 0.2
RANDOM_STATE = 42


@dataclass
class PreprocessedData:
    """Train/test features, targets, and the fitted preprocessing pipeline."""

    X_train: Any
    X_test: Any
    y_train: pd.Series
    y_test: pd.Series
    pipeline: Pipeline
    numeric_columns: list[str]
    categorical_columns: list[str]
    original_feature_names: list[str]
    engineered_feature_names: list[str]
    feature_names: list[str]
    removed_columns: list[str]


def load_raw_dataset(data_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load a CSV without changing its source file."""
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Loan dataset not found: {path}")
    return pd.read_csv(path)


def split_features_target(
    dataset: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    id_columns: Sequence[str] = NON_PREDICTIVE_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Separate labels and exclude identifiers that cannot describe applicants."""
    if target_column not in dataset.columns:
        raise ValueError(f"Target column {target_column!r} is missing from the dataset.")
    if dataset[target_column].isna().any():
        raise ValueError(f"Target column {target_column!r} contains missing values.")

    # Loan_ID only distinguishes records; learning its arbitrary label would not
    # provide applicant information and could encourage memorizing applications.
    removed_columns = [column for column in id_columns if column in dataset.columns]
    X = dataset.drop(columns=[target_column, *removed_columns]).copy()
    y = dataset[target_column].copy()
    if X.empty:
        raise ValueError("No predictive features remain after removing the target and IDs.")
    return X, y, removed_columns


def identify_feature_columns(
    features: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Classify numeric measurements separately from categorical inputs."""
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]
    if not numeric_columns and not categorical_columns:
        raise ValueError("No feature columns are available for preprocessing.")
    return numeric_columns, categorical_columns


def build_preprocessing_pipeline(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> Pipeline:
    """Create one fitted-at-training-time transformer for later inference."""
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, list(numeric_columns)))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, list(categorical_columns))
        )

    if not transformers:
        raise ValueError("At least one numeric or categorical feature is required.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("feature_engineering", LoanFeatureEngineer()),
            ("preprocessor", preprocessor),
        ]
    )


def prepare_data(
    data_path: str | Path = DEFAULT_DATA_PATH,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> PreprocessedData:
    """Split raw data, fit transformations only on training rows, and transform both."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    dataset = load_raw_dataset(data_path)
    X, y, removed_columns = split_features_target(dataset, target_column)
    # Inspect only an empty frame to learn the output schema; feature values are
    # created separately inside the pipeline after the train/test split.
    engineered_schema = engineer_features(X.head(0))
    numeric_columns, categorical_columns = identify_feature_columns(engineered_schema)

    # Split before fitting imputers, the encoder, or the scaler to prevent test
    # observations from influencing any learned preprocessing statistics.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    pipeline = build_preprocessing_pipeline(numeric_columns, categorical_columns)
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    engineered_feature_names = (
        pipeline.named_steps["feature_engineering"].get_feature_names_out().tolist()
    )

    if X_train_transformed.shape[1] != len(feature_names):
        raise RuntimeError("Transformed feature names do not match the feature matrix.")
    if X_test_transformed.shape[1] != len(feature_names):
        raise RuntimeError("Train and test transformations produced different features.")

    return PreprocessedData(
        X_train=X_train_transformed,
        X_test=X_test_transformed,
        y_train=y_train,
        y_test=y_test,
        pipeline=pipeline,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        original_feature_names=X.columns.tolist(),
        engineered_feature_names=engineered_feature_names,
        feature_names=feature_names,
        removed_columns=removed_columns,
    )


def _assert_finite(matrix: Any, name: str) -> None:
    """Fail clearly if any missing or infinite values survive preprocessing."""
    values = matrix.data if hasattr(matrix, "tocsr") else np.asarray(matrix)
    if not np.isfinite(values).all():
        raise AssertionError(f"{name} contains missing or non-finite transformed values.")


def main() -> None:
    """Run the preprocessing workflow and report its validation checks."""
    dataset = load_raw_dataset()
    result = prepare_data()
    _assert_finite(result.X_train, "Training features")
    _assert_finite(result.X_test, "Testing features")

    feature_engineer = result.pipeline.named_steps["feature_engineering"]
    preprocessor = result.pipeline.named_steps["preprocessor"]
    if "Loan_Status" in result.original_feature_names:
        raise AssertionError("The target must not be included in feature engineering.")
    if "Loan_Status" in result.engineered_feature_names:
        raise AssertionError("Feature engineering must not create or use the target.")
    numeric_transformer = preprocessor.named_transformers_.get("numeric")
    if numeric_transformer is not None:
        numeric_feature_names = [
            f"numeric__{column}" for column in result.numeric_columns
        ]
        numeric_indices = [
            result.feature_names.index(name) for name in numeric_feature_names
        ]
        numeric_train = result.X_train[:, numeric_indices]
        numeric_values = (
            numeric_train.toarray()
            if hasattr(numeric_train, "toarray")
            else np.asarray(numeric_train)
        )
        if not np.allclose(numeric_values.mean(axis=0), 0, atol=1e-9):
            raise AssertionError("Scaled training numeric features should have mean zero.")
        if not np.allclose(numeric_values.std(axis=0), 1, atol=1e-9):
            raise AssertionError("Scaled training numeric features should have unit variance.")

    categorical_feature_names = [
        name for name in result.feature_names if name.startswith("categorical__")
    ]
    if result.categorical_columns:
        categorical_transformer = preprocessor.named_transformers_["categorical"]
        encoder = categorical_transformer.named_steps["encoder"]
        expected_encoded_columns = sum(len(categories) for categories in encoder.categories_)
        if len(categorical_feature_names) != expected_encoded_columns:
            raise AssertionError("One-hot encoded columns do not match learned categories.")

    print(f"Raw dataset: {dataset.shape[0]} rows x {dataset.shape[1]} columns")
    print(f"Removed non-predictive columns: {result.removed_columns}")
    print(f"Predictive features before engineering: {result.original_feature_names}")
    print(
        "Predictive features after engineering: "
        f"{feature_engineer.get_feature_names_out().tolist()}"
    )
    print(
        f"Predictive inputs: {len(result.numeric_columns)} numeric, "
        f"{len(result.categorical_columns)} categorical"
    )
    print(
        "Train/test rows before transformation: "
        f"{result.X_train.shape[0]} / {result.X_test.shape[0]}"
    )
    print(
        f"Split: train {result.X_train.shape[0]} rows, "
        f"test {result.X_test.shape[0]} rows"
    )
    print(
        f"Transformed features: train {result.X_train.shape[0]} x "
        f"{result.X_train.shape[1]}, test {result.X_test.shape[0]} x "
        f"{result.X_test.shape[1]}"
    )
    print(
        f"Training target rows: {len(result.y_train)}; "
        f"testing target rows: {len(result.y_test)}"
    )
    print("Validation passed: train and test contain only finite transformed values.")
    print("Validation passed: numeric training features have zero mean and unit variance.")
    print(
        f"Validation passed: {len(categorical_feature_names)} one-hot encoded "
        "categorical output columns are present."
    )
    print("Validation passed: the same fitted pipeline transformed train and test data.")


if __name__ == "__main__":
    main()
