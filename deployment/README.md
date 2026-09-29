# Deployment

## Recommended portfolio architecture

- **React + Vite frontend:** Vercel static hosting. Set the project root to
  `frontend`, build with `npm run build`, and publish `dist`. The included
  `frontend/vercel.json` sends client-side paths such as `/history` to the SPA.
- **FastAPI API:** Render Python web service using the root `render.yaml`
  Blueprint. It installs `backend/requirements.txt`, starts Uvicorn on Render's
  assigned port, and checks `/health`.
- **MySQL:** Aiven managed MySQL. Create the service and database without
  replacing or resetting any existing data. Apply `database/schema.sql` only
  when setting up the new database or when its required tables are absent.
- **ML inference:** The Render service loads the already packaged
  `ml/models/loan_risk_pipeline.joblib` from the repository. Deployment does
  not train or modify the model. The source training CSV is kept for the
  documented ML workflow and tests.

The included provider configurations describe deployment, but this project
does not deploy automatically. Do not configure a Git remote or trigger an
external deployment without authorization.

## Backend environment

Set these variables in the Render service dashboard; the Blueprint declares
the keys without embedding values:

| Variable | Purpose |
| --- | --- |
| `DB_HOST` | Aiven MySQL host |
| `DB_PORT` | Aiven MySQL port |
| `DB_NAME` | Existing application database name |
| `DB_USER` | Database user with the required database privileges |
| `DB_PASSWORD` | Database password, entered only in the hosting dashboard |
| `DB_SSL_CA` | Path to the Aiven CA certificate mounted as a Render secret file |
| `CORS_ORIGINS` | Comma-separated exact HTTPS origin(s) of the deployed frontend |

For Aiven, download its CA certificate from the service console and add it as a
Render secret file. Set `DB_SSL_CA` to that mounted file's path. SQLAlchemy's
PyMySQL connection verifies the certificate and server identity when this
setting is present. Never commit the certificate if the provider treats it as
sensitive, and never put database credentials in source or build logs.

For local development, the root `.env.example` contains variable names and
placeholders only. Copy it to `.env`, replace the placeholders locally, and
keep `.env` untracked. Production credentials belong in the Render dashboard,
not in a deployed `.env` file.

## Frontend environment

Set `VITE_API_BASE_URL` in the Vercel project's build environment to the public
HTTPS origin of the Render API (for example, `https://your-api.example.com`).
Vite embeds `VITE_*` values into the static bundle at build time, so changing
the value requires a new frontend build/deployment. The tracked
`frontend/env.production.example` is a placeholder template only. Set the same
API origin as the sole or one of the comma-separated values in Render's
`CORS_ORIGINS`.

The development API default is limited to Vite development. A production build
without `VITE_API_BASE_URL` does not silently call localhost; API requests show
a user-safe configuration message.

## Deployment sequence

1. Create an Aiven MySQL service and database. Configure TLS and the application
   account in the provider console. Back up existing databases before any
   operational schema changes.
2. Create the Render web service from this repository and review `render.yaml`.
   Set all database values, CA secret-file path, and the exact deployed
   frontend origin in Render's dashboard.
3. Confirm the Render `/health` endpoint responds successfully. Verify the
   configured MySQL user can access the target database and that
   `loan_applications`, `predictions`, and their foreign key already exist.
4. Create the Vercel project with `frontend` as its root directory. Set
   `VITE_API_BASE_URL` to the Render API origin and deploy the static build.
5. From the deployed frontend, verify a prediction and history retrieval. Check
   that both saved rows are linked and the history is newest-first. Do not reset
   or drop production data for verification.

## Local production-build check

In PowerShell, build with a non-secret API URL to check production bundling:

```powershell
$env:VITE_API_BASE_URL = "https://your-api.example.com"
Set-Location frontend
npm run build
Remove-Item Env:VITE_API_BASE_URL
```

Local builds verify frontend compilation only; they do not establish external
provider availability, TLS setup, deployed CORS behavior, or public DNS.

## Manual provider setup required

Deployment requires user-controlled Render, Vercel, and Aiven accounts; an
authorized source repository connection; service creation; provider-specific
host/user/password values; Aiven CA certificate provisioning; frontend/API
public origins; and any required billing or region choices. These actions are
not performed by this repository configuration.
