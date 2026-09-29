"""Leakage-safe, row-wise feature engineering for loan applications."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


ENGINEERED_FEATURES = (
    "TotalIncome",
    "LoanAmountToIncome",
    "LoanTermYears",
    "DependentsNumeric",
)
REQUIRED_SOURCE_COLUMNS = (
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Dependents",
)
NON_FEATURE_COLUMNS = frozenset({"Loan_ID", "Loan_Status"})
DEPENDENT_COUNTS = {"0": 0, "1": 1, "2": 2, "3+": 3}


def engineer_features(features: pd.DataFrame) -> pd.DataFrame:
    """Return applicant features with four deterministic domain-driven additions.

    ``Dependents`` is replaced by its numeric lower-bound representation to avoid
    redundantly feeding the same field to both numeric and categorical encoders.
    Missing values are preserved for the downstream training-fitted imputers.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("Feature engineering expects a pandas DataFrame.")
    forbidden = NON_FEATURE_COLUMNS.intersection(features.columns)
    if forbidden:
        raise ValueError(
            "Feature engineering must not receive target or identifier columns: "
            f"{sorted(forbidden)}"
        )
    missing_columns = [
        column for column in REQUIRED_SOURCE_COLUMNS if column not in features.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Required source columns are missing: {missing_columns}"
        )

    engineered = features.copy()
    total_income = (
        pd.to_numeric(features["ApplicantIncome"], errors="raise")
        + pd.to_numeric(features["CoapplicantIncome"], errors="raise")
    )
    loan_amount = pd.to_numeric(features["LoanAmount"], errors="raise")
    loan_term_months = pd.to_numeric(
        features["Loan_Amount_Term"], errors="raise"
    )

    # A missing/non-positive income cannot form a meaningful denominator; leave
    # its ratio missing so the training-fitted numeric imputer handles it.
    positive_income = total_income.where(total_income > 0)
    engineered["TotalIncome"] = total_income
    engineered["LoanAmountToIncome"] = loan_amount.div(positive_income)
    engineered["LoanTermYears"] = loan_term_months.div(12)

    dependents = features["Dependents"].astype("string")
    dependents_numeric = dependents.map(DEPENDENT_COUNTS).astype("float64")
    unknown_dependents = features["Dependents"].notna() & dependents_numeric.isna()
    if unknown_dependents.any():
        unexpected = features.loc[unknown_dependents, "Dependents"].unique().tolist()
        raise ValueError(f"Unexpected Dependents categories: {unexpected}")
    engineered["DependentsNumeric"] = dependents_numeric
    return engineered.drop(columns="Dependents")


class LoanFeatureEngineer(TransformerMixin, BaseEstimator):
    """Sklearn-compatible, stateless row-wise feature transformer."""

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> LoanFeatureEngineer:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("LoanFeatureEngineer expects a pandas DataFrame.")
        # Validate the schema and forbidden columns without learning data statistics.
        engineer_features(X.head(0))
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["feature_names_in_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError("LoanFeatureEngineer expects a pandas DataFrame.")
        if list(X.columns) != self.feature_names_in_.tolist():
            raise ValueError(
                "Input columns and order must match the columns observed during fit."
            )
        return engineer_features(X)

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        check_is_fitted(self, attributes=["feature_names_in_"])
        if input_features is not None and list(input_features) != self.feature_names_in_.tolist():
            raise ValueError("input_features must match the fitted input columns.")
        output_features = [
            column
            for column in self.feature_names_in_.tolist()
            if column != "Dependents"
        ]
        output_features.extend(ENGINEERED_FEATURES)
        return np.asarray(output_features, dtype=object)
