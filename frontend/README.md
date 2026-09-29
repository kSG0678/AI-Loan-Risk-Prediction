# Loan Prediction Frontend

A responsive React interface for sending applicant details to the existing
FastAPI `POST /predict` endpoint and reviewing persisted results from
`GET /predictions/history`. The frontend displays the model's class label and
both probabilities. It does not train a model or implement prediction logic.

## Start the application

Start FastAPI from the project root in one terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

In another terminal, install frontend packages and start Vite:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (by default, `http://127.0.0.1:5173`).
The API client calls `http://127.0.0.1:8000` by default for prediction and
history requests. To use a different API origin, set `VITE_API_BASE_URL` in a
frontend-local `.env` file, for example:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

The default Vite port is included in the backend's development CORS allowlist.

## Request flow

1. The applicant form collects the eleven fields accepted by FastAPI.
2. The API client converts numeric form values to JSON numbers and sends them
   to `POST /predict`.
3. The existing backend validates the request and invokes the saved ML pipeline.
4. The page displays the returned predicted class and approval/rejection
   probabilities, or a readable error if the API cannot respond.
5. Use the Prediction history navigation to view saved records from the backend.
   The history page preserves the API's newest-first order and provides loading,
   empty, and safe error states.

The form starts with an editable example application. Predictions are estimates
for educational use only, not lending decisions or financial advice.

## Build for a production bundle

```powershell
npm run build
npm run preview
```
