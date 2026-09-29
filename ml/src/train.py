"""Train reproducible baseline classifiers on the shared loan preprocessing.

Run ``.venv/Scripts/python.exe ml/src/train.py`` from the project root. Models,
test predictions, and probability outputs remain in memory for Phase 6.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

if __package__:
    from .data_preprocessing import PreprocessedData, prepare_data
else:
    from data_preprocessing import PreprocessedData, prepare_data


RANDOM_STATE = 42


@dataclass
class ModelRun:
    """One fitted classifier and its held-out predictions and timing."""

    model: ClassifierMixin
    predictions: np.ndarray
    probabilities: np.ndarray
    positive_class_probability: np.ndarray
    accuracy: float
    training_seconds: float


@dataclass
class TrainingResults:
    """In-memory Phase 5 results, including the common fitted preprocessor."""

    preprocessing_pipeline: Any
    data: PreprocessedData
    model_runs: dict[str, ModelRun]


def build_model_configurations() -> dict[str, ClassifierMixin]:
    """Create comparable baseline classifiers with reproducible settings."""
    return {
        # Linear classifier: learns a weighted, additive decision boundary.
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        # Single tree: learns readable if/then splits and captures non-linear rules.
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        # Tree ensemble: averages many randomized trees to reduce single-tree variance.
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        # Kernel classifier: learns a margin between classes. Cross-validated
        # calibration adds probabilities without relying on deprecated SVC behavior.
        "Support Vector Machine": CalibratedClassifierCV(
            estimator=SVC(random_state=RANDOM_STATE),
            method="sigmoid",
            cv=5,
            ensemble=False,
        ),
        # Instance-based classifier: predicts from nearby training applications.
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        # Probabilistic baseline: applies Bayes' rule with a Gaussian numeric model.
        "Gaussian Naive Bayes": GaussianNB(),
    }


def _as_model_input(matrix: Any, model_name: str) -> Any:
    """Convert sparse encoded features only for estimators that require dense input."""
    if model_name == "Gaussian Naive Bayes" and hasattr(matrix, "toarray"):
        return matrix.toarray()
    return matrix


def train_models(data: PreprocessedData | None = None) -> TrainingResults:
    """Fit each classifier on the same train matrix and predict the same test rows."""
    prepared = data if data is not None else prepare_data()
    configurations = build_model_configurations()
    model_runs: dict[str, ModelRun] = {}

    # Feature engineering and preprocessing were fitted on training rows only.
    # Reusing the exact matrices gives every classifier identical inputs and splits.
    for model_name, model in configurations.items():
        X_train = _as_model_input(prepared.X_train, model_name)
        X_test = _as_model_input(prepared.X_test, model_name)

        started = perf_counter()
        model.fit(X_train, prepared.y_train)
        training_seconds = perf_counter() - started

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        positive_class_index = np.flatnonzero(model.classes_ == "Y")
        if len(positive_class_index) != 1:
            raise ValueError(
                f"{model_name} did not expose exactly one approval ('Y') class."
            )
        positive_class_probability = probabilities[:, positive_class_index[0]]

        model_runs[model_name] = ModelRun(
            model=model,
            predictions=predictions,
            probabilities=probabilities,
            positive_class_probability=positive_class_probability,
            accuracy=float(accuracy_score(prepared.y_test, predictions)),
            training_seconds=training_seconds,
        )

    return TrainingResults(
        preprocessing_pipeline=prepared.pipeline,
        data=prepared,
        model_runs=model_runs,
    )


def main() -> None:
    """Train all baselines and print concise held-out prediction summaries."""
    results = train_models()
    data = results.data
    print(
        f"Shared split: X_train={data.X_train.shape}, X_test={data.X_test.shape}, "
        f"y_train={data.y_train.shape}, y_test={data.y_test.shape}"
    )
    print(
        "Shared preprocessing pipeline: feature engineering, train-fitted "
        "imputation/scaling/one-hot encoding."
    )
    print()

    for model_name, run in results.model_runs.items():
        predicted_counts = {
            str(label): int(count)
            for label, count in zip(*np.unique(run.predictions, return_counts=True))
        }
        print(
            f"{model_name}: trained successfully; "
            f"test accuracy={run.accuracy:.3f}; "
            f"training time={run.training_seconds:.4f}s; "
            f"predictions={predicted_counts}; "
            f"probabilities shape={run.probabilities.shape}"
        )

    print(
        f"\nAll {len(results.model_runs)} models trained and predicted successfully. "
        "Fitted models and prediction outputs remain in memory for Phase 6."
    )
    print("No final model has been selected.")


if __name__ == "__main__":
    main()
