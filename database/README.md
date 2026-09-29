# MySQL Database

The MySQL schema stores validated loan inputs in `loan_applications` and
prediction outcomes in `predictions`. Each prediction references one
application. The existing `/predict` endpoint now writes both records after
successful inference.

## Create the database and tables

Create the database using a MySQL account allowed to create databases:

```sql
CREATE DATABASE loan_risk_db
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;
```

Then select `loan_risk_db` and execute [`schema.sql`](./schema.sql). For example,
from PowerShell with the MySQL command-line client installed:

```powershell
cmd.exe /c "mysql -u root -p loan_risk_db < database\schema.sql"
```

MySQL 8.0.16 or later is recommended so the schema's `CHECK` constraints are
enforced.

If you choose a different database name, create and select that database
instead; the schema script creates tables in the currently selected database.

## Backend environment

Set these variables in the environment where the backend runs:

| Variable | Purpose | Example |
| --- | --- | --- |
| `DB_HOST` | MySQL server hostname | `127.0.0.1` |
| `DB_PORT` | MySQL server port | `3306` |
| `DB_NAME` | Database containing the tables | `loan_risk_db` |
| `DB_USER` | MySQL account with access to the database | `loan_app` |
| `DB_PASSWORD` | Password for that account | Set privately, never commit it |

The backend loads the project-root `.env` file when its database configuration
is imported. Existing process environment variables take precedence. Keep `.env`
excluded from version control and never commit actual credentials. In
PowerShell, values for the current session can also be set like this:

```powershell
$env:DB_HOST = "127.0.0.1"
$env:DB_PORT = "3306"
$env:DB_NAME = "loan_risk_db"
$env:DB_USER = "loan_app"
$securePassword = Read-Host "MySQL password" -AsSecureString
$env:DB_PASSWORD = [System.Net.NetworkCredential]::new("", $securePassword).Password
```

## Verify the database connection

Install the backend dependencies, then from the project root run:

```powershell
.\.venv\Scripts\python.exe -c "from backend.database import check_database_connection, create_database_engine; engine = create_database_engine(); check_database_connection(engine); print('MySQL connection successful'); engine.dispose()"
```

This explicitly attempts a real database connection and fails with the
underlying SQLAlchemy/driver error if configuration or connectivity is invalid.
It is separate from starting the API, which does not connect to MySQL.

## Prediction persistence

For each valid request, inference runs first. If it succeeds, the backend
inserts the application and its linked prediction in one SQLAlchemy transaction.
Both rows commit together; if either insert or the commit fails, the transaction
is rolled back and the API returns a generic HTTP `500` persistence error without
returning SQL, credentials, or stack details. The engine and session factory
are shared; a request gets its own session, and no database connection is
opened at application startup.

## Verify a saved prediction

After configuring MySQL and making a successful `POST /predict`, inspect the
newest records with:

```sql
SELECT
    a.id AS application_id,
    a.gender,
    a.applicant_income,
    a.created_at AS application_created_at,
    p.id AS prediction_id,
    p.predicted_class,
    p.approval_probability,
    p.rejection_probability,
    p.created_at AS prediction_created_at
FROM loan_applications AS a
JOIN predictions AS p ON p.application_id = a.id
ORDER BY p.id DESC
LIMIT 10;
```

## Run persistence tests

From the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest backend.tests.test_api
.\.venv\Scripts\python.exe -m unittest backend.tests.test_database
```

The API tests override the production session dependency with an isolated
in-memory SQLite database, including rollback checks. They do not require a
running MySQL server.
