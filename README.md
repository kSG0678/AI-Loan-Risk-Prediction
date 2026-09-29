# AI Loan Risk & Approval Platform

An educational full-stack project for exploring loan risk prediction with a
saved scikit-learn pipeline, FastAPI, MySQL, and a React frontend.

## Project status

The model inference pipeline, prediction API, transactional MySQL persistence,
read-only prediction history API, and React prediction/history pages are
implemented.

## Project areas

- `ml/` — data exploration, preprocessing, model training, evaluation, and
  saved-model inference.
- `backend/` — FastAPI routes, request/response schemas, persistence models,
  and tests.
- `database/` — MySQL schema and setup instructions.
- `frontend/` — React application for new assessments and saved prediction
  history.

## Setup

Clone the repository and open its project directory:

```powershell
git clone <repository-url> AI-Loan-Risk-Prediction
cd AI-Loan-Risk-Prediction
```

Use Python 3.13 and Node.js/npm. Create the Python virtual environment and
install the ML and backend dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r ml\requirements.txt
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

Create a local root `.env` from the template, then replace its placeholder
values locally:

```powershell
Copy-Item .env.example .env
```

Never commit `.env` or put real credentials in `.env.example`.

Create the MySQL database and apply [`database/schema.sql`](database/schema.sql)
as described in [`database/README.md`](database/README.md). The backend loads
the root `.env`; the file is excluded by `.gitignore` and must never be
committed.

## Run locally

Start the API from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

In another terminal, install frontend packages and start Vite:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (normally `http://127.0.0.1:5173`). The API
documentation is available at `http://127.0.0.1:8000/docs`.

## API

- `POST /predict` validates an applicant, runs the existing saved model, and
  transactionally saves the application and prediction.
- `GET /predictions/history` reads saved application and prediction details,
  newest prediction first.
- `GET /health` provides a lightweight API health check.

The request and response examples, database setup, and endpoint details are in
[`backend/README.md`](backend/README.md) and
[`database/README.md`](database/README.md). The frontend links to prediction
history at `/history`; its setup and behavior are described in
[`frontend/README.md`](frontend/README.md).

## Verification

Run the backend API, database, and ML prediction tests from the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest backend.tests.test_api
.\.venv\Scripts\python.exe -m unittest backend.tests.test_database
.\.venv\Scripts\python.exe -m unittest ml.src.test_prediction
.\.venv\Scripts\python.exe -m compileall -q backend ml
.\.venv\Scripts\python.exe -m pip check
```

Build the frontend from its directory:

```powershell
cd frontend
npm run build
```
