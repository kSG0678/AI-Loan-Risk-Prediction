const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

export async function requestPrediction(applicant) {
  let response;

  try {
    response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(applicant),
    });
  } catch {
    throw new Error(
      "Could not reach the prediction API. Check that FastAPI is running at " +
        `${API_BASE_URL} and try again.`,
    );
  }

  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const details = Array.isArray(body?.details)
      ? body.details
          .map((item) => {
            const field = item.field?.split(".").at(-1);
            return field ? `${field}: ${item.message}` : item.message;
          })
          .join(" ")
      : body?.message;

    throw new Error(details || body?.error || "The prediction could not be completed.");
  }

  return body;
}

export async function requestPredictionHistory() {
  let response;

  try {
    response = await fetch(`${API_BASE_URL}/predictions/history`);
  } catch {
    throw new Error("We couldn’t connect to the prediction service. Please try again.");
  }

  if (!response.ok) {
    throw new Error("Prediction history is temporarily unavailable. Please try again.");
  }

  try {
    const records = await response.json();
    if (!Array.isArray(records)) {
      throw new Error("Invalid history response");
    }
    return records;
  } catch {
    throw new Error("Prediction history is temporarily unavailable. Please try again.");
  }
}
