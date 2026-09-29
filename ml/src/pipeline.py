"""Build, save, and load the complete loan approval inference pipeline.

Run as a package module: ``python -m ml.src.pipeline`` from the project root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

if __package__:
    from .data_preprocessing import PROJECT_ROOT, prepare_data
else:
    from data_preprocessing import PROJECT_ROOT, prepare_data


MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "loan_risk_pipeline.joblib"
RANDOM_STATE = 42


def build_inference_pipeline() -> Pipeline:
    """Fit the selected candidate on train data and assemble all inference steps."""
    prepared = prepare_data()
    classifier = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    # Reuse the fitted training-only transformers returned by Phase 3, then fit
    # the candidate on that same training split; the held-out test set is unused.
    classifier.fit(prepared.X_train, prepared.y_train)
    return Pipeline(
        steps=[
            (
                "feature_engineering",
                prepared.pipeline.named_steps["feature_engineering"],
            ),
            ("preprocessor", prepared.pipeline.named_steps["preprocessor"]),
            ("classifier", classifier),
        ]
    )


def save_inference_pipeline(
    pipeline: Pipeline, artifact_path: str | Path = MODEL_PATH
) -> Path:
    """Persist the entire fitted inference pipeline, not only its classifier."""
    destination = Path(artifact_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination, compress=3)
    return destination


def load_inference_pipeline(
    artifact_path: str | Path = MODEL_PATH,
) -> Pipeline:
    """Load a previously packaged pipeline without fitting or training anything."""
    source = Path(artifact_path)
    if not source.is_file():
        raise FileNotFoundError(f"Packaged inference pipeline not found: {source}")
    loaded: Any = joblib.load(source)
    if not isinstance(loaded, Pipeline):
        raise TypeError(f"Expected a scikit-learn Pipeline in {source}.")
    expected_steps = ["feature_engineering", "preprocessor", "classifier"]
    if [name for name, _ in loaded.steps] != expected_steps:
        raise ValueError(
            f"Unexpected pipeline steps in {source}: "
            f"{[name for name, _ in loaded.steps]}"
        )
    return loaded


def package_candidate(
    artifact_path: str | Path = MODEL_PATH,
) -> Pipeline:
    """Train and save the Phase 6 metric-based candidate for later inference."""
    pipeline = build_inference_pipeline()
    destination = save_inference_pipeline(pipeline, artifact_path)
    print(f"Saved complete Logistic Regression inference pipeline: {destination}")
    print(f"Artifact size: {destination.stat().st_size:,} bytes")
    print("Pipeline steps: feature engineering -> preprocessing -> classifier")
    print("Training used the Phase 5 training split only; no final model claim is made.")
    return pipeline


def main() -> None:
    package_candidate()


if __name__ == "__main__":
    main()
