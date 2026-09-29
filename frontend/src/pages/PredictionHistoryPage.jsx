import { useEffect, useState } from "react";
import SiteHeader from "../components/SiteHeader.jsx";
import { requestPredictionHistory } from "../services/predictionApi.js";

const moneyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
});

const dateFormatter = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short",
});

function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Date unavailable" : dateFormatter.format(date);
}

function HistoryRecord({ record }) {
  return (
    <article className="history-card">
      <div className="history-card-heading">
        <div>
          <span className="eyebrow">APPLICATION #{record.application_id}</span>
          <h2>Prediction #{record.prediction_id}</h2>
        </div>
        <span
          className={`history-class ${record.predicted_class === "Y" ? "is-approved" : "is-rejected"}`}
        >
          {record.predicted_class === "Y" ? "Approved" : "Rejected"} ({record.predicted_class})
        </span>
      </div>

      <dl className="history-details">
        <div>
          <dt>Applicant income</dt>
          <dd>{moneyFormatter.format(record.applicant_income)}</dd>
        </div>
        <div>
          <dt>Co-applicant income</dt>
          <dd>{moneyFormatter.format(record.coapplicant_income)}</dd>
        </div>
        <div>
          <dt>Loan amount</dt>
          <dd>{moneyFormatter.format(record.loan_amount)}</dd>
        </div>
        <div>
          <dt>Loan term</dt>
          <dd>{record.loan_amount_term} months</dd>
        </div>
        <div>
          <dt>Credit history</dt>
          <dd>{record.credit_history === 1 ? "Meets guidelines" : "Does not meet guidelines"}</dd>
        </div>
        <div>
          <dt>Education</dt>
          <dd>{record.education}</dd>
        </div>
        <div>
          <dt>Property area</dt>
          <dd>{record.property_area}</dd>
        </div>
        <div>
          <dt>Created at</dt>
          <dd>{formatDate(record.created_at)}</dd>
        </div>
      </dl>

      <div className="history-probabilities" aria-label="Prediction probabilities">
        <div>
          <span>Approval probability</span>
          <strong>{(record.approval_probability * 100).toFixed(2)}%</strong>
        </div>
        <div>
          <span>Rejection probability</span>
          <strong>{(record.rejection_probability * 100).toFixed(2)}%</strong>
        </div>
      </div>
    </article>
  );
}

export default function PredictionHistoryPage() {
  const [records, setRecords] = useState([]);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isCurrent = true;

    requestPredictionHistory()
      .then((history) => {
        if (isCurrent) {
          setRecords(history);
        }
      })
      .catch(() => {
        if (isCurrent) {
          setError("We couldn’t load prediction history right now. Please try again.");
        }
      })
      .finally(() => {
        if (isCurrent) {
          setIsLoading(false);
        }
      });

    return () => {
      isCurrent = false;
    };
  }, []);

  return (
    <div className="app-shell">
      <SiteHeader activePage="history" />
      <main className="history-main">
        <section className="history-intro">
          <div className="breadcrumb">
            <span>PERSONAL FINANCE</span>
            <span className="breadcrumb-slash">/</span>
            <span>PREDICTION HISTORY</span>
          </div>
          <h1>Your saved <span>predictions.</span></h1>
          <p>Review applicant details and results from previous assessments.</p>
        </section>

        {isLoading ? (
          <section className="history-state" aria-live="polite">
            <span className="spinner history-spinner" aria-hidden="true" />
            <h2>Loading prediction history</h2>
            <p>Retrieving your saved assessments.</p>
          </section>
        ) : error ? (
          <section className="history-state history-error" role="alert">
            <h2>History is unavailable</h2>
            <p>{error}</p>
            <button
              className="button button-primary"
              type="button"
              onClick={() => window.location.reload()}
            >
              Try again
            </button>
          </section>
        ) : records.length === 0 ? (
          <section className="history-state" aria-live="polite">
            <h2>No predictions yet</h2>
            <p>Completed assessments will appear here.</p>
            <a className="button button-primary history-link-button" href="/">
              Start an assessment
            </a>
          </section>
        ) : (
          <section className="history-list" aria-label="Saved prediction history">
            {records.map((record) => (
              <HistoryRecord key={record.prediction_id} record={record} />
            ))}
          </section>
        )}
      </main>
      <footer className="footer">
        <span>northstar<span className="footer-period">.</span> <span className="footer-year">LOAN INSIGHTS</span></span>
        <span>Educational estimate only · Not financial advice</span>
      </footer>
    </div>
  );
}
