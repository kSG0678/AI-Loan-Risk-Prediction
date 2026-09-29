# Machine Learning

This area contains the loan dataset and exploratory analysis. Later phases can add reusable preprocessing and feature-engineering code, model training and evaluation, saved model metadata, and generated reports. The original dataset belongs under `data/raw/` and must remain unchanged.

## Phase 2: dataset and exploratory analysis

`data/raw/train_u6lujuX_CVtuZ9i.csv` is the labeled training split from the Analytics Vidhya Loan Prediction practice problem. It is stored as downloaded; exploratory analysis reads it without changing it. The dataset contains applicant and co-applicant details, loan information, and a `Loan_Status` approval outcome. Its source is the [Analytics Vidhya practice problem](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction/) and this [commit-pinned public mirror](https://github.com/Dhakal29/LoanPrediction/blob/e3122c7d230504ddbd8b4ad9bad1b1e60ae60d63/train_u6lujuX_CVtuZ9i.csv).

Run `notebooks/01_eda.ipynb` from the project root or its notebook directory. The notebook requires Python with `pandas`, `numpy`, `matplotlib`, and `seaborn` installed. It inspects the raw data and produces descriptive statistics and visualizations only; it does not preprocess data or train a model.

## Python environment

Use Python 3.13 for the project environment. The repository's `.vscode/settings.json` points the VS Code Python extension to `.venv` and enables terminal activation. From the project root, create the environment and install the pinned ML and notebook dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r ml\requirements.txt
```

The virtual environment is local to this project and excluded from version control. Keeping the interpreter and compiled scientific packages in the project environment avoids mixing them with global or user-site Python packages.

## Phase 3: data preprocessing

Run the reusable, leakage-safe preprocessing workflow from the project root using the project environment:

```powershell
.\.venv\Scripts\python.exe ml\src\data_preprocessing.py
```

The module reads the unchanged CSV in `data/raw/`, separates `Loan_Status` as the target, and excludes `Loan_ID` from the predictive features because it is an arbitrary application identifier rather than applicant or loan information. The input CSV is never overwritten. No processed CSV is saved: the fitted sklearn pipeline and transformed train/test matrices are returned in memory so later work can reuse the exact transformations without creating a redundant dataset.

### Workflow and concepts

1. **Separate inputs and target:** `Loan_Status` is held apart from applicant features. `Loan_ID` is removed because its unique labels do not describe an applicant and should not be learned as a predictor.
2. **Identify feature types:** numeric columns are selected by their data type; the remaining applicant fields are treated as categorical.
3. **Split before fitting:** `train_test_split` divides features and target into a reproducible 80% training set and 20% test set (`random_state=42`, stratified by approval outcome). The test split represents unseen examples for checking the preparation workflow. Stratification keeps the approval proportions similar in both partitions.
4. **Impute missing values:** `SimpleImputer` fills missing numeric values with the training-set median, which is less affected by extreme values than the mean. A separate `SimpleImputer` fills missing categorical values with the most frequent training value, preserving a valid category without inventing a new one. Because the imputers are inside the pipeline and are fitted after splitting, test-set statistics cannot leak into training.
5. **Encode categorical values:** `OneHotEncoder` turns categories such as property area into separate 0/1 indicator columns. This lets a model use category membership without implying that category labels have a numeric order. `handle_unknown="ignore"` allows inference to continue if a later application contains a category absent during training.
6. **Scale numeric values:** `StandardScaler` centers each numeric input around the training mean and scales it to unit variance. This puts measurements with different units and ranges on comparable scales; the scaler learns these statistics from training rows only.
7. **Combine transformations:** `ColumnTransformer` routes numeric and categorical columns through their respective steps and combines their outputs, while dropping no additional columns.
8. **Reuse one pipeline:** the sklearn `Pipeline` keeps the imputation, scaling, and encoding steps together. It is fitted only with training data, then the same fitted pipeline transforms both training and test features; the object can later be reused on individual inference inputs.

The module checks that both transformed partitions contain finite values, the scaled training numeric features have zero mean and unit variance, and categorical output columns were created. It does not train or compare any models.

## Phase 4: feature engineering

Feature engineering is implemented in `src/feature_engineering.py` as a stateless, reusable sklearn transformer. It is the first step in the preprocessing pipeline, before imputation, scaling, and encoding. The transformation is applied row by row, learns no statistics from either split, and never receives `Loan_Status` or `Loan_ID`. The existing train/test split and downstream transformers are unchanged; their statistics continue to be fitted on training rows only.

The raw predictive feature list, after excluding `Loan_ID` and `Loan_Status`, is:

`Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`

The engineered feature list is:

`Gender`, `Married`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`, `TotalIncome`, `LoanAmountToIncome`, `LoanTermYears`, `DependentsNumeric`

The original `Dependents` category is replaced by `DependentsNumeric`; keeping both would encode the same information twice.

| Engineered feature | Meaning and calculation | Why it may help | Assumptions and limitations |
|---|---|---|---|
| `TotalIncome` | Applicant income plus co-applicant income. | Represents the combined income associated with an application; the source columns are also retained so a later model can distinguish their contributions. | Assumes the two recorded amounts are comparable and should be considered together. It may remain strongly skewed; this phase does not clip or log-transform values. |
| `LoanAmountToIncome` | `LoanAmount / TotalIncome`; a non-positive total income or missing loan amount produces a missing ratio for the existing training-fitted imputer. | Adds a relative loan-size measure alongside absolute income and loan amount. | The dataset does not document the time basis or exact unit scale of these fields, so this is a rough relative ratio, not a verified debt-to-income or affordability measure. Extreme values remain possible. |
| `LoanTermYears` | `Loan_Amount_Term / 12`. | Expresses the recorded term in familiar years rather than months. | Assumes the source term is in months; it is a unit conversion, not new information. |
| `DependentsNumeric` | Maps `0`, `1`, and `2` to their counts and `3+` to `3`, the bucket's lower bound; missing values remain missing for imputation. | Allows numeric transformations and models to use the ordered household-size signal. | Treats the open-ended `3+` bucket as 3, losing differences among larger households; the numeric representation also assumes an ordered, roughly monotonic effect. The original category is removed to avoid duplicate encoding. |

Use `LoanFeatureEngineer` directly for a DataFrame of applicant fields, or use the complete fitted pipeline returned by `prepare_data()` for both feature engineering and preprocessing. The complete pipeline is the preferred inference interface so new applications receive exactly the same transformations. No target-derived features, full-dataset statistics, processed files, or model training are introduced in this phase.

## Phase 5: baseline model training

Run the six reproducible supervised-learning baselines from the project root:

```powershell
.\.venv\Scripts\python.exe ml\src\train.py
```

`src/train.py` calls the Phase 3 `prepare_data()` workflow once. The same stratified 80/20 split (491 training and 123 test applications with `random_state=42`) and the same already-fitted feature-engineering and preprocessing outputs are then used for all six estimators. The imputer, scaler, and encoder are fitted only on the training split; the held-out test data is transformed but never used to fit preprocessing or classifiers. This separation helps avoid optimistic results from information leaking from evaluation examples into training.

### Algorithms included

| Algorithm | Why it is included |
|---|---|
| Logistic Regression | A simple linear classification baseline that learns weighted feature contributions and returns probabilities. |
| Decision Tree Classifier | Learns non-linear if/then feature splits and offers a tree-based contrast to a linear model. |
| Random Forest Classifier | Combines randomized decision trees to provide an ensemble baseline that is less dependent on one tree. Its inclusion does not mean it is automatically the best model. |
| Support Vector Machine (`SVC`) | Learns a maximum-margin boundary and can represent non-linear boundaries with its default kernel. A sigmoid `CalibratedClassifierCV` fitted with training-only cross-validation provides probabilities for later evaluation without relying on the deprecated `SVC(probability=True)` option. |
| K-Nearest Neighbors | An instance-based baseline that predicts from nearby training examples; feature scaling is important for its distance calculations. |
| Gaussian Naive Bayes | A fast probabilistic baseline that applies Bayes' rule with a conditional Gaussian assumption for numeric features. |

Each classifier receives the exact same transformed training and test matrices. Gaussian Naive Bayes requires dense input, so only its copy of the already-transformed sparse matrix is converted to dense; its values, features, and rows are unchanged. The fitted classifier objects, test predictions, class probabilities, approval probabilities, test accuracy, and training time are kept in the `TrainingResults` returned by `train_models()`; nothing is persisted as a model artifact in this phase.

Reproducibility means fixing the split seed and model seeds where supported, using the same data and transformations, and declaring estimator settings so a rerun under the same software environment can repeat the experiment. Training times may vary with hardware and system load. The printed accuracy and predictions are basic training-phase outputs only; the full Phase 6 evaluation is documented below.

## Phase 6: model evaluation and selection

Run the evaluation from the project root:

```powershell
.\.venv\Scripts\python.exe ml\src\evaluate.py
```

The evaluator reruns the fixed Phase 5 training procedure and calculates accuracy, approval-class (`Y`) precision, recall, F1-score, ROC-AUC, and a confusion matrix for every classifier on the same 123-row stratified test split. ROC-AUC uses approval probabilities; the SVC probabilities come from sigmoid calibration fit with cross-validation on training data. The held-out test set is not used to fit preprocessing, models, or the SVC calibration.

The run writes:

- `reports/model_comparison.csv` — metrics and confusion-matrix counts for all models.
- `reports/model_comparison.png` — grouped metric comparison.
- `reports/confusion_matrices/` — one CSV per model and a combined heatmap.
- `reports/roc_curves/roc_curves.png` — ROC curves for all six models.
- `reports/model_evaluation.md` — dataset details, all metrics, confusion matrices, ROC-AUC comparison, metric definitions, trade-offs, and selection rationale.

For the model-selection comparison, accuracy is shown but is not used as the sole or ranking criterion. Models are ranked equally across precision, recall, F1-score, and ROC-AUC; mean rank is the objective comparison, with ties broken by F1-score and then ROC-AUC. The highest-ranked model is only a metric-based candidate: the small, single historical holdout and absence of a stated business preference for false positives versus false negatives limit any production claim. This phase does not save a fitted model.

## Phase 7: model packaging and inference

Package the Phase 6 Logistic Regression candidate from the project root:

```powershell
.\.venv\Scripts\python.exe -m ml.src.pipeline
```

This trains only the selected candidate on the reproducible 491-row training split (`random_state=42`) and saves the complete fitted inference pipeline to `ml/models/loan_risk_pipeline.joblib`. The artifact contains, in order:

1. `LoanFeatureEngineer` for deterministic, row-wise loan features.
2. The fitted `ColumnTransformer` containing training-fitted numeric and categorical imputers, numeric scaler, and one-hot encoder.
3. The fitted Logistic Regression classifier.

`Loan_Status` is separated from applicant features and is never part of inference input. `Loan_ID` is also excluded because it identifies an application rather than describing an applicant. The raw dataset remains unchanged. The held-out test split is not used to fit the packaged transformers or classifier.

Training and inference are distinct: packaging fits transformations and the candidate on the training data; inference loads those fitted objects and predicts without training or reading the dataset. Keeping feature engineering, imputers, scaling, encoding, and the estimator together prevents serving code from implementing preprocessing differently from training.

Use the reusable prediction function with one mapping containing the original 11 applicant feature fields (do not include `Loan_ID` or `Loan_Status`):

```python
from ml.src.predict import predict_applicant

applicant = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "1",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 4583,
    "CoapplicantIncome": 1508,
    "LoanAmount": 128,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Semiurban",
}
result = predict_applicant(applicant)
```

`predict_applicant` loads the joblib artifact, validates the fields, applies packaged feature engineering and preprocessing, and returns the predicted class (`Y`/`N`), human-readable label, approval probability, and rejection probability. Missing feature values can use the saved imputers; categories unseen during training are handled by the configured one-hot encoder. `Loan_ID` and `Loan_Status` are rejected as inputs. A future FastAPI service can load the artifact once at startup and call `predict_with_pipeline` for validated requests instead of retraining or loading the file for every request.

Verify packaging and prediction from the project root:

```powershell
.\.venv\Scripts\python.exe -m ml.src.pipeline
.\.venv\Scripts\python.exe -m unittest ml.src.test_prediction
```

This is only the candidate selected by Phase 6's documented metric-ranking rule on a small, single historical test split. Packaging it does not make it production-ready; no API or deployment is included in this phase.
