"""Evaluate the six baseline loan classifiers and write comparison reports.

Run ``.venv/Scripts/python.exe ml/src/evaluate.py`` from the project root.
This reruns the reproducible Phase 5 training workflow; it does not persist a
production model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

if __package__:
    from .data_preprocessing import PROJECT_ROOT
    from .train import TrainingResults, train_models
else:
    from data_preprocessing import PROJECT_ROOT
    from train import TrainingResults, train_models


REPORTS_DIR = PROJECT_ROOT / "ml" / "reports"
LABELS = ["N", "Y"]
POSITIVE_LABEL = "Y"
METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
SELECTION_METRICS = ["Precision", "Recall", "F1-score", "ROC-AUC"]


@dataclass
class EvaluationResults:
    """Metric table, confusion matrices, and the training artifacts evaluated."""

    metrics: pd.DataFrame
    confusion_matrices: dict[str, np.ndarray]
    training_results: TrainingResults
    selected_candidate: str
    selection_ranks: pd.DataFrame


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def evaluate_models(
    training_results: TrainingResults | None = None,
) -> EvaluationResults:
    """Calculate held-out metrics for each Phase 5 model using approval as positive."""
    training = training_results if training_results is not None else train_models()
    y_true = training.data.y_test.to_numpy()
    if not set(np.unique(y_true)).issubset(set(LABELS)):
        raise ValueError(f"Expected target labels to be drawn from {LABELS}.")

    metric_rows: list[dict[str, Any]] = []
    matrices: dict[str, np.ndarray] = {}

    for model_name, run in training.model_runs.items():
        predictions = np.asarray(run.predictions)
        approval_probabilities = np.asarray(run.positive_class_probability)
        if predictions.shape != y_true.shape:
            raise ValueError(
                f"{model_name} has {len(predictions)} predictions for "
                f"{len(y_true)} test labels."
            )
        if approval_probabilities.shape != y_true.shape:
            raise ValueError(
                f"{model_name} has invalid approval-probability dimensions."
            )
        if not np.isfinite(approval_probabilities).all():
            raise ValueError(f"{model_name} produced non-finite ROC-AUC scores.")

        matrix = confusion_matrix(y_true, predictions, labels=LABELS)
        if matrix.shape != (2, 2):
            raise RuntimeError(
                f"{model_name} confusion matrix must be 2x2, got {matrix.shape}."
            )
        true_negative, false_positive, false_negative, true_positive = (
            int(value) for value in matrix.ravel()
        )
        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_true, predictions),
            "Precision": precision_score(
                y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0
            ),
            "Recall": recall_score(
                y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0
            ),
            "F1-score": f1_score(
                y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(
                (y_true == POSITIVE_LABEL).astype(int), approval_probabilities
            ),
            "TN": true_negative,
            "FP": false_positive,
            "FN": false_negative,
            "TP": true_positive,
        }
        metric_rows.append(metrics)
        matrices[model_name] = matrix

    metrics_frame = pd.DataFrame(metric_rows).set_index("Model")
    metric_values = metrics_frame[METRIC_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(metric_values).all():
        raise RuntimeError("One or more evaluation metrics are non-finite.")
    if ((metric_values < 0) | (metric_values > 1)).any():
        raise RuntimeError("Evaluation metrics must fall between 0 and 1.")

    # Give equal weight to each non-accuracy metric; accuracy is reported but does
    # not influence the selection, so a high majority-class score alone cannot win.
    rank_frame = metrics_frame[SELECTION_METRICS].rank(
        ascending=False, method="average"
    )
    rank_frame["MeanRank"] = rank_frame.mean(axis=1)
    rank_frame = rank_frame.sort_values(
        ["MeanRank", "F1-score", "ROC-AUC"], ascending=[True, False, False]
    )
    selected_candidate = str(rank_frame.index[0])

    return EvaluationResults(
        metrics=metrics_frame,
        confusion_matrices=matrices,
        training_results=training,
        selected_candidate=selected_candidate,
        selection_ranks=rank_frame,
    )


def _write_confusion_matrix_artifacts(results: EvaluationResults) -> None:
    output_dir = REPORTS_DIR / "confusion_matrices"
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name, matrix in results.confusion_matrices.items():
        matrix_frame = pd.DataFrame(
            matrix,
            index=pd.Index(LABELS, name="Actual"),
            columns=pd.Index(LABELS, name="Predicted"),
        )
        matrix_frame.to_csv(output_dir / f"{_slugify(model_name)}.csv")

    model_names = list(results.confusion_matrices)
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, model_name in zip(axes.flat, model_names):
        sns.heatmap(
            results.confusion_matrices[model_name],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=LABELS,
            yticklabels=LABELS,
            ax=axis,
        )
        axis.set_title(model_name)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
    figure.suptitle("Confusion matrices (N = not approved, Y = approved)")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrices.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def _write_roc_curve(results: EvaluationResults) -> None:
    output_dir = REPORTS_DIR / "roc_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = (
        results.training_results.data.y_test.to_numpy() == POSITIVE_LABEL
    ).astype(int)
    figure, axis = plt.subplots(figsize=(8, 7))
    for model_name, run in results.training_results.model_runs.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true, run.positive_class_probability
        )
        auc = results.metrics.loc[model_name, "ROC-AUC"]
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{model_name} (AUC={auc:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set(
        title="ROC curves for loan approval prediction",
        xlabel="False-positive rate",
        ylabel="True-positive rate (recall)",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.legend(loc="lower right", fontsize="small")
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curves.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def _write_metric_comparison(results: EvaluationResults) -> None:
    ordered = results.metrics.sort_values("F1-score", ascending=False)
    figure, axis = plt.subplots(figsize=(12, 6))
    ordered[METRIC_COLUMNS].plot.bar(ax=axis)
    axis.set(
        title="Model comparison on the shared test set",
        xlabel="Model",
        ylabel="Score (0 to 1)",
        ylim=(0, 1.05),
    )
    axis.tick_params(axis="x", rotation=25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(
        REPORTS_DIR / "model_comparison.png", dpi=160, bbox_inches="tight"
    )
    plt.close(figure)


def _format_confusion_matrix(matrix: np.ndarray) -> str:
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return (
        "| Actual / Predicted | N (not approved) | Y (approved) |\n"
        "|---|---:|---:|\n"
        f"| N (not approved) | {tn} (TN) | {fp} (FP) |\n"
        f"| Y (approved) | {fn} (FN) | {tp} (TP) |"
    )


def _markdown_table(frame: pd.DataFrame, float_format: str = ".3f") -> str:
    """Format a small report table without an extra tabulate dependency."""
    columns = [str(column) for column in frame.columns]
    rows = []
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(format(float(value), float_format))
            else:
                values.append(str(value))
        rows.append(values)
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _build_report(results: EvaluationResults) -> str:
    data = results.training_results.data
    metrics_table = results.metrics.reset_index()
    metrics_markdown = _markdown_table(metrics_table)
    auc_comparison = (
        results.metrics[["ROC-AUC"]]
        .sort_values("ROC-AUC", ascending=False)
        .reset_index()
    )
    auc_comparison_markdown = _markdown_table(auc_comparison)
    confusion_sections = []
    for model_name, matrix in results.confusion_matrices.items():
        confusion_sections.append(
            f"### {model_name}\n\n{_format_confusion_matrix(matrix)}"
        )

    return f"""# Model Evaluation and Comparison

## Dataset and test-set information

- Raw labeled dataset: **614 rows × 13 columns**.
- Split: **{len(data.y_train)} training rows** and **{len(data.y_test)} held-out test rows** (stratified, `random_state=42`).
- Transformed input dimensions: **{data.X_train.shape[1]} features** for each training row and **{data.X_test.shape[1]} features** for each test row.
- Test labels: {int((data.y_test == "N").sum())} not approved (`N`) and {int((data.y_test == "Y").sum())} approved (`Y`).
- Positive class for precision, recall, F1, and ROC-AUC: **approved (`Y`)**.
- The train-only fitted feature-engineering and preprocessing pipeline and the same split were shared across all six models. No test data was used to fit the pipeline or classifiers.

## Metrics table

{metrics_markdown}

All metrics are calculated from the same held-out test labels and predictions. Precision, recall, and F1 treat approval (`Y`) as the positive class. Confusion-matrix columns in the table count true negatives (`TN`), false positives (`FP`), false negatives (`FN`), and true positives (`TP`).

## Confusion matrices

Rows are actual outcomes; columns are predicted outcomes. `N` means not approved and `Y` means approved.

{chr(10).join(confusion_sections)}

Combined visualization: `confusion_matrices/confusion_matrices.png`. Individual matrix data is stored as CSV files in `confusion_matrices/`.

## ROC-AUC comparison

{auc_comparison_markdown}

ROC curves: `roc_curves/roc_curves.png`. ROC-AUC uses each model's approval (`Y`) probability. The SVC probability estimates come from sigmoid calibration fitted with cross-validation on training data only. ROC-AUC measures how well a model ranks positive cases above negative cases across possible score thresholds; it does not choose an operational threshold.

## Metric meanings and model trade-offs

- **True Positive (TP):** an application that was approved in the data and predicted approved.
- **True Negative (TN):** an application that was not approved and predicted not approved.
- **False Positive (FP):** an application predicted approved that was not approved in the historical labels. In this prediction task, it is an incorrect approval prediction.
- **False Negative (FN):** an application predicted not approved that was approved in the historical labels. It is a missed approval prediction.
- **Precision:** among applications predicted approved, the share that were actually approved (`TP / (TP + FP)`). Higher precision means fewer false approval predictions among predicted approvals.
- **Recall:** among actually approved applications, the share predicted approved (`TP / (TP + FN)`). Higher recall means fewer missed approvals.
- **F1-score:** the harmonic mean of precision and recall; it is high when both are high and penalizes a large imbalance.
- **ROC-AUC:** the probability-based ranking measure described above; 0.5 is chance ranking and 1.0 is perfect separation on the evaluated sample.

False positives and false negatives represent different prediction mistakes. Their business impact depends on lender policy, costs, regulation, and applicant outcomes. No business priority or relative cost has been provided, so this report does not claim one error type is more important.

The six algorithms have different inductive assumptions: Logistic Regression learns an additive linear boundary; a Decision Tree forms sequential rules; Random Forest averages randomized trees; SVC learns a margin boundary (with probabilities calibrated for this evaluation); K-Nearest Neighbors relies on distances to training observations; Gaussian Naive Bayes estimates class probabilities under a conditional-independence/Gaussian assumption. These differences can shift the precision/recall balance and probability ranking even when accuracy is similar.

## Production-model selection rationale

**Metric-based candidate: {results.selected_candidate}.** The selection rule is an equal-weight average rank across **precision, recall, F1-score, and ROC-AUC**. For each of these four metrics, rank all six models from highest to lowest; the model with the lowest mean rank is the balanced metric leader. Ties are broken by higher F1-score and then higher ROC-AUC. Accuracy is reported for transparency but is not part of this selection rule, so the choice is not based on accuracy alone.

| Model | Precision rank | Recall rank | F1 rank | ROC-AUC rank | Mean rank |
|---|---:|---:|---:|---:|---:|
{chr(10).join(f"| {name} | {row['Precision']:.1f} | {row['Recall']:.1f} | {row['F1-score']:.1f} | {row['ROC-AUC']:.1f} | {row['MeanRank']:.3f} |" for name, row in results.selection_ranks.iterrows())}

This is a metric-based candidate for subsequent review, **not a claim of production readiness**. The test set has only {len(data.y_test)} applications, is a single holdout, and reflects a historical practice dataset. There is no declared business error-cost preference or independent validation set. Phase 7 can package a model only after the project owner reviews these limitations and the metric trade-offs. No model has been saved or deployed.
"""


def write_reports(results: EvaluationResults) -> None:
    """Write CSV, figures, and the generated Markdown evaluation report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.metrics.reset_index().to_csv(
        REPORTS_DIR / "model_comparison.csv", index=False, float_format="%.6f"
    )
    _write_confusion_matrix_artifacts(results)
    _write_roc_curve(results)
    _write_metric_comparison(results)
    (REPORTS_DIR / "model_evaluation.md").write_text(
        _build_report(results), encoding="utf-8"
    )


def main() -> None:
    """Train the common baselines, evaluate them, and save comparison artifacts."""
    results = evaluate_models()
    write_reports(results)
    print("Evaluation metrics (positive class: Y = approved):")
    print(results.metrics.to_string(float_format=lambda value: f"{value:.3f}"))
    print("\nConfusion matrices (actual rows N,Y; predicted columns N,Y):")
    for model_name, matrix in results.confusion_matrices.items():
        print(f"\n{model_name}\n{matrix}")
    print(f"\nROC-AUC comparison:\n{results.metrics['ROC-AUC'].sort_values(ascending=False).to_string(float_format=lambda value: f'{value:.3f}')}")
    print(f"\nMetric-based candidate: {results.selected_candidate}")
    print(f"Reports written to {REPORTS_DIR}")
    print("No production model was persisted.")


if __name__ == "__main__":
    main()
