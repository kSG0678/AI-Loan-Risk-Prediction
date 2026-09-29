import { useState } from "react";
import ApplicantForm from "../components/ApplicantForm.jsx";
import PredictionResult from "../components/PredictionResult.jsx";
import SiteHeader from "../components/SiteHeader.jsx";
import { requestPrediction } from "../services/predictionApi.js";
import {
  INITIAL_APPLICANT,
  toApplicantPayload,
} from "../utils/applicantFields.js";

export default function PredictionPage() {
  const [values, setValues] = useState(INITIAL_APPLICANT);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  function handleChange(event) {
    const { name, value } = event.target;
    setValues((current) => ({ ...current, [name]: value }));
    setResult(null);
    setError("");
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setResult(null);
    setIsSubmitting(true);

    try {
      const prediction = await requestPrediction(toApplicantPayload(values));
      setResult(prediction);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsSubmitting(false);
    }
  }

  function handleReset() {
    setValues(INITIAL_APPLICANT);
    setResult(null);
    setError("");
  }

  return (
    <div className="app-shell">
      <SiteHeader activePage="assessment" />

      <main>
        <section className="intro">
          <div className="intro-copy">
            <div className="breadcrumb">
              <span>PERSONAL FINANCE</span>
              <span className="breadcrumb-slash">/</span>
              <span>LOAN ASSESSMENT</span>
            </div>
            <h1>A clearer view<br />of your <span>next step.</span></h1>
            <p>
              Share a few details to see an AI-generated estimate of your loan
              application. It only takes a moment.
            </p>
          </div>
          <div className="intro-stamp" aria-label="Simple, private, transparent">
            <span className="stamp-star" aria-hidden="true">✳</span>
            <span>Simple.</span>
            <span>Private.</span>
            <span>Transparent.</span>
          </div>
        </section>

        <section className="workspace" aria-label="Loan prediction assessment">
          <div className="form-card">
            <div className="card-topline">
              <div>
                <span className="eyebrow">APPLICATION DETAILS</span>
                <h2>Tell us about your application</h2>
              </div>
              <div className="required-note"><span>*</span> All fields required</div>
            </div>
            <ApplicantForm
              values={values}
              onChange={handleChange}
              onSubmit={handleSubmit}
              onReset={handleReset}
              isSubmitting={isSubmitting}
            />
          </div>
          <PredictionResult
            result={result}
            error={error}
            isSubmitting={isSubmitting}
          />
        </section>

        <section className="privacy-strip" aria-label="Privacy information">
          <span className="privacy-icon" aria-hidden="true">
            <svg viewBox="0 0 20 20" fill="none">
              <rect x="3.5" y="8" width="13" height="9" rx="2" stroke="currentColor" strokeWidth="1.4" />
              <path d="M6.5 8V5.5a3.5 3.5 0 0 1 7 0V8" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
            </svg>
          </span>
          <p>
            <strong>Your information stays in this session.</strong> Details are
            sent to the local prediction API and are not stored by this page.
          </p>
          <span className="privacy-caption">DEMO EXPERIENCE</span>
        </section>
      </main>

      <footer className="footer">
        <span>northstar<span className="footer-period">.</span> <span className="footer-year">LOAN INSIGHTS</span></span>
        <span>Educational estimate only · Not financial advice</span>
      </footer>
    </div>
  );
}
