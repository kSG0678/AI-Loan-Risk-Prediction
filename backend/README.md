# FastAPI Backend

This backend exposes the project's existing loan-risk inference pipeline over
HTTP. FastAPI receives JSON, Pydantic checks the applicant fields, and the
prediction service uses `ml/models/loan_risk_pipeline.joblib` to return a
prediction. The backend neither trains a model nor reads the raw training data
when serving requests.

## Folder responsibilities

- `main.py` creates the FastAPI application, configures local development CORS,
  registers clean error responses, and includes the API router.
- `api/routes.py` defines the root, health, and prediction endpoints.
- `schemas/loan_schema.py` validates the eleven applicant fields and documents
  the prediction response shape.
- `services/prediction_service.py` loads and caches the saved pipeline, then
  delegates prediction to `ml/src/predict.py`.
- `utils/config.py` resolves the saved artifact path and lists allowed local
  frontend origins.
- `database/config.py` creates a lazy SQLAlchemy engine from environment
  variables and provides connection/session helpers.
- `models/` defines the loan-application and prediction persistence models.
- `tests/test_api.py` makes HTTP requests to the FastAPI application and checks
  success, validation, caching, and no-training behavior.
- `tests/test_database.py` validates the models and environment-based
  configuration with SQLite; its tests do not require a MySQL server.

## Request lifecycle

```text
User
  ↓
POST /predict
  ↓
Pydantic validation of the applicant JSON
  ↓
Prediction service obtains the cached saved pipeline
  ↓
ml/src/predict.py creates the input DataFrame and invokes that pipeline
  ↓
The saved feature engineering, preprocessing, and classifier produce a result
  ↓
A request-scoped SQLAlchemy session inserts the application and linked
prediction in one transaction
  ↓
FastAPI returns the prediction and probabilities as JSON
```

Separating HTTP routes, validation schemas, and the prediction service keeps
each part focused: routes handle web requests, schemas reject invalid data,
and the service connects the existing ML code to the API.

## Endpoints

- `GET /` returns the API's online status.
- `GET /health` returns `{"status": "healthy"}`.
- `POST /predict` validates one applicant and returns the predicted class,
  display label, approval probability, and rejection probability.
- `GET /predictions/history` reads saved applicant details and prediction
  results from MySQL, ordered by newest prediction first.

### Prediction history

Request the persisted history with:

```http
GET /predictions/history
```

The endpoint only reads the existing `loan_applications` and `predictions`
tables; it does not run inference, train the model, or access the training
dataset. An empty history is returned as an empty JSON array.

Example response:

```json
[
  {
    "application_id": 1,
    "prediction_id": 1,
    "applicant_income": 4583.0,
    "coapplicant_income": 1508.0,
    "loan_amount": 128.0,
    "loan_amount_term": 360.0,
    "credit_history": 1,
    "education": "Graduate",
    "property_area": "Semiurban",
    "predicted_class": "Y",
    "approval_probability": 0.89,
    "rejection_probability": 0.11,
    "created_at": "2026-09-29T12:00:00"
  }
]
```

Database query failures return HTTP `500` with a generic error; SQL and internal
exception details are not included in the response.

## Request and response formats

Send all eleven fields using the categories learned by the saved pipeline:

```json
{
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
  "Property_Area": "Semiurban"
}
```

The successful response has this shape; prediction values are calculated by
the saved model and vary with the applicant:

```json
{
  "predicted_class": "Y",
  "predicted_class_label": "Approved",
  "approval_probability": 0.89,
  "rejection_probability": 0.11
}
```

Invalid or missing fields return HTTP `422` with a JSON error and field details.
Model loading failures return HTTP `503`; prediction failures return HTTP `500`.
Internal exception details are logged by the backend and are not sent to clients.

## How the model is loaded

The service locates the existing `ml/models/loan_risk_pipeline.joblib` file
relative to the project folder and loads it with the existing
`ml/src/pipeline.py` loader. It is loaded lazily on the first prediction and
cached in memory, so later requests reuse the same fitted feature engineering,
preprocessing, and classifier steps rather than rereading the file. The backend
does not invoke training functions or load the raw dataset.

## Prediction persistence

After inference succeeds, `/predict` maps the already validated
`ApplicantRequest` into a `loan_applications` row and stores the output in a
linked `predictions` row. Both inserts occur in the same SQLAlchemy transaction:
the API commits only when both records are saved, and a failure rolls the
transaction back so neither row remains. Database failures return HTTP `500`
with a generic persistence error; SQL, credentials, and exception details are
kept out of the API response.

The database engine and session factory are cached and shared by the backend.
The engine is created lazily on first database use, so application startup does
not connect to MySQL. The backend loads the project-root `.env` file when its
database configuration is imported; existing process environment variables
take precedence. Configure `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, and
`DB_PASSWORD` before serving requests. See
[`database/README.md`](../database/README.md) for schema creation, configuration,
and a manual query to verify saved rows.

## Start the server

From the project root, install backend dependencies into the existing virtual
environment and run Uvicorn:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation. The
development CORS list allows local frontend origins on ports `3000` and `5173`;
production origins are intentionally not enabled.

Example request using curl:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
  -H 'Content-Type: application/json' `
  -d '{"Gender":"Male","Married":"Yes","Dependents":"1","Education":"Graduate","Self_Employed":"No","ApplicantIncome":4583,"CoapplicantIncome":1508,"LoanAmount":128,"Loan_Amount_Term":360,"Credit_History":1,"Property_Area":"Semiurban"}'
```

## Run tests

Run backend HTTP tests and the existing saved-pipeline prediction tests from the
project root:

```powershell
.\.venv\Scripts\python.exe -m unittest backend.tests.test_api
.\.venv\Scripts\python.exe -m unittest backend.tests.test_database
.\.venv\Scripts\python.exe -m unittest ml.src.test_prediction
```

The API tests override the database session with an isolated in-memory SQLite
database; the database tests also use SQLite. Neither test suite needs a live
MySQL server.

## Database setup

The persistence layer uses SQLAlchemy with the PyMySQL driver. Install
`backend/requirements.txt`, create the `loan_risk_db` database, and run
`database/schema.sql` against it. Configure `DB_HOST`, `DB_PORT`, `DB_NAME`,
`DB_USER`, and `DB_PASSWORD` in the backend process environment. Detailed
database setup and a real connection-check command are documented in
[`database/README.md`](../database/README.md). MySQL is not contacted at API
startup.

Check the installed Python dependency consistency with:

```powershell
.\.venv\Scripts\python.exe -m pip check
```
