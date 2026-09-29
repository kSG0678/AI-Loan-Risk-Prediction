function SparkleIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 2.75 14.2 9.8l7.05 2.2-7.05 2.2L12 21.25 9.8 14.2l-7.05-2.2L9.8 9.8 12 2.75Z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <path d="m19 2 .7 2.3L22 5l-2.3.7L19 8l-.7-2.3L16 5l2.3-.7L19 2Z" fill="currentColor" />
    </svg>
  );
}

function ProbabilityRow({ label, probability, tone }) {
  const percentage = Math.round(probability * 100);

  return (
    <div className="probability-row">
      <div className="probability-copy">
        <span className={`probability-dot ${tone}`} />
        <span>{label}</span>
        <strong>{percentage}%</strong>
      </div>
      <div
        className="probability-track"
        role="progressbar"
        aria-label={`${label} probability`}
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow={percentage}
      >
        <span
          className={`probability-fill ${tone}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

export default function PredictionResult({ result, error, isSubmitting }) {
  if (isSubmitting) {
    return (
      <aside className="result-panel result-pending" aria-live="polite">
        <div className="result-icon"><SparkleIcon /></div>
        <span className="eyebrow">YOUR ASSESSMENT</span>
        <h2>Reviewing your details</h2>
        <p>The saved prediction pipeline is processing your application.</p>
        <div className="pending-lines" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
      </aside>
    );
  }

  if (error) {
    return (
      <aside className="result-panel result-error" aria-live="assertive">
        <div className="result-icon error-icon" aria-hidden="true">!</div>
        <span className="eyebrow">ASSESSMENT UNAVAILABLE</span>
        <h2>We couldn’t complete this request</h2>
        <p>{error}</p>
        <div className="result-footnote">
          Your information has not been saved. Please review the form and try again.
        </div>
      </aside>
    );
  }

  if (!result) {
    return (
      <aside className="result-panel result-empty" aria-live="polite">
        <div className="result-icon"><SparkleIcon /></div>
        <span className="eyebrow">YOUR ASSESSMENT</span>
        <h2>Your result, made clearer.</h2>
        <p>
          Complete the applicant details and request an assessment to see the
          model’s prediction.
        </p>
        <div className="empty-divider" />
        <div className="empty-note">
          <span className="note-check" aria-hidden="true">✓</span>
          <span>Both approval and rejection probabilities are shown.</span>
        </div>
        <div className="empty-note">
          <span className="note-check" aria-hidden="true">✓</span>
          <span>Your result comes from the existing saved model.</span>
        </div>
      </aside>
    );
  }

  const approved = result.predicted_class === "Y";

  return (
    <aside className={`result-panel result-complete ${approved ? "is-approved" : "is-rejected"}`} aria-live="polite">
      <div className="result-icon"><SparkleIcon /></div>
      <span className="eyebrow">YOUR ASSESSMENT</span>
      <div className={`result-status ${approved ? "status-approved" : "status-rejected"}`}>
        <span className="status-dot" />
        Model prediction
      </div>
      <h2>{result.predicted_class_label}</h2>
      <p className="result-summary">
        {approved
          ? "Based on the details provided, the model predicts this application may be approved."
          : "Based on the details provided, the model predicts this application may be rejected."}
      </p>

      <div className="confidence-card">
        <div className="confidence-heading">
          <span>Prediction probabilities</span>
          <span className="confidence-caption">MODEL OUTPUT</span>
        </div>
        <ProbabilityRow
          label="Approval"
          probability={result.approval_probability}
          tone="approval"
        />
        <ProbabilityRow
          label="Rejection"
          probability={result.rejection_probability}
          tone="rejection"
        />
      </div>

      <div className="result-footnote">
        A model estimate for learning and demonstration—not a guarantee or
        official lending decision.
      </div>
    </aside>
  );
}
