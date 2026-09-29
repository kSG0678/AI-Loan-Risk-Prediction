# Model Evaluation and Comparison

## Dataset and test-set information

- Raw labeled dataset: **614 rows × 13 columns**.
- Split: **491 training rows** and **123 held-out test rows** (stratified, `random_state=42`).
- Transformed input dimensions: **20 features** for each training row and **20 features** for each test row.
- Test labels: 38 not approved (`N`) and 85 approved (`Y`).
- Positive class for precision, recall, F1, and ROC-AUC: **approved (`Y`)**.
- The train-only fitted feature-engineering and preprocessing pipeline and the same split were shared across all six models. No test data was used to fit the pipeline or classifiers.

## Metrics table

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.862 | 0.840 | 0.988 | 0.908 | 0.870 | 22 | 16 | 1 | 84 |
| Decision Tree | 0.683 | 0.811 | 0.706 | 0.755 | 0.669 | 24 | 14 | 25 | 60 |
| Random Forest | 0.846 | 0.875 | 0.906 | 0.890 | 0.860 | 27 | 11 | 8 | 77 |
| Support Vector Machine | 0.846 | 0.830 | 0.976 | 0.897 | 0.863 | 21 | 17 | 2 | 83 |
| K-Nearest Neighbors | 0.862 | 0.870 | 0.941 | 0.904 | 0.838 | 26 | 12 | 5 | 80 |
| Gaussian Naive Bayes | 0.821 | 0.832 | 0.929 | 0.878 | 0.822 | 22 | 16 | 6 | 79 |

All metrics are calculated from the same held-out test labels and predictions. Precision, recall, and F1 treat approval (`Y`) as the positive class. Confusion-matrix columns in the table count true negatives (`TN`), false positives (`FP`), false negatives (`FN`), and true positives (`TP`).

## Confusion matrices

Rows are actual outcomes; columns are predicted outcomes. `N` means not approved and `Y` means approved.

### Logistic Regression

| Actual / Predicted | N (not approved) | Y (approved) |
|---|---:|---:|
| N (not approved) | 22 (TN) | 16 (FP) |
| Y (approved) | 1 (FN) | 84 (TP) |
### Decision Tree

| Actual / Predicted | N (not approved) | Y (approved) |
|---|---:|---:|
| N (not approved) | 24 (TN) | 14 (FP) |
| Y (approved) | 25 (FN) | 60 (TP) |
### Random Forest

| Actual / Predicted | N (not approved) | Y (approved) |
|---|---:|---:|
| N (not approved) | 27 (TN) | 11 (FP) |
| Y (approved) | 8 (FN) | 77 (TP) |
### Support Vector Machine

| Actual / Predicted | N (not approved) | Y (approved) |
|---|---:|---:|
| N (not approved) | 21 (TN) | 17 (FP) |
| Y (approved) | 2 (FN) | 83 (TP) |
### K-Nearest Neighbors

| Actual / Predicted | N (not approved) | Y (approved) |
|---|---:|---:|
| N (not approved) | 26 (TN) | 12 (FP) |
| Y (approved) | 5 (FN) | 80 (TP) |
### Gaussian Naive Bayes

| Actual / Predicted | N (not approved) | Y (approved) |
|---|---:|---:|
| N (not approved) | 22 (TN) | 16 (FP) |
| Y (approved) | 6 (FN) | 79 (TP) |

Combined visualization: `confusion_matrices/confusion_matrices.png`. Individual matrix data is stored as CSV files in `confusion_matrices/`.

## ROC-AUC comparison

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.870 |
| Support Vector Machine | 0.863 |
| Random Forest | 0.860 |
| K-Nearest Neighbors | 0.838 |
| Gaussian Naive Bayes | 0.822 |
| Decision Tree | 0.669 |

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

**Metric-based candidate: Logistic Regression.** The selection rule is an equal-weight average rank across **precision, recall, F1-score, and ROC-AUC**. For each of these four metrics, rank all six models from highest to lowest; the model with the lowest mean rank is the balanced metric leader. Ties are broken by higher F1-score and then higher ROC-AUC. Accuracy is reported for transparency but is not part of this selection rule, so the choice is not based on accuracy alone.

| Model | Precision rank | Recall rank | F1 rank | ROC-AUC rank | Mean rank |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 3.0 | 1.0 | 1.0 | 1.0 | 1.500 |
| K-Nearest Neighbors | 2.0 | 3.0 | 2.0 | 4.0 | 2.750 |
| Support Vector Machine | 5.0 | 2.0 | 3.0 | 2.0 | 3.000 |
| Random Forest | 1.0 | 5.0 | 4.0 | 3.0 | 3.250 |
| Gaussian Naive Bayes | 4.0 | 4.0 | 5.0 | 5.0 | 4.500 |
| Decision Tree | 6.0 | 6.0 | 6.0 | 6.0 | 6.000 |

This is a metric-based candidate for subsequent review, **not a claim of production readiness**. The test set has only 123 applications, is a single holdout, and reflects a historical practice dataset. There is no declared business error-cost preference or independent validation set. Phase 7 can package a model only after the project owner reviews these limitations and the metric trade-offs. No model has been saved or deployed.
