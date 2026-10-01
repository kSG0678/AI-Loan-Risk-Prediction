export const INITIAL_APPLICANT = {
  Gender: "",
  Married: "",
  Dependents: "",
  Education: "",
  Self_Employed: "",
  ApplicantIncome: "",
  CoapplicantIncome: "",
  LoanAmount: "",
  Loan_Amount_Term: "",
  Credit_History: "",
  Property_Area: "",
};

export const APPLICANT_FIELDS = [
  {
    name: "Gender",
    label: "Gender",
    type: "select",
    options: ["Female", "Male"],
  },
  {
    name: "Married",
    label: "Marital status",
    type: "select",
    options: ["Yes", "No"],
  },
  {
    name: "Dependents",
    label: "Number of dependents",
    type: "select",
    options: ["0", "1", "2", "3+"],
  },
  {
    name: "Education",
    label: "Education",
    type: "select",
    options: ["Graduate", "Not Graduate"],
  },
  {
    name: "Self_Employed",
    label: "Self-employed",
    type: "select",
    options: ["No", "Yes"],
  },
  {
    name: "ApplicantIncome",
    label: "Your monthly income",
    type: "number",
    min: "0",
    step: "any",
    prefix: "$",
  },
  {
    name: "CoapplicantIncome",
    label: "Co-applicant monthly income",
    type: "number",
    min: "0",
    step: "any",
    prefix: "$",
  },
  {
    name: "LoanAmount",
    label: "Requested loan amount",
    type: "number",
    min: "0",
    step: "any",
    prefix: "$",
    hint: "Enter the amount in thousands, as used by the model.",
  },
  {
    name: "Loan_Amount_Term",
    label: "Loan term",
    type: "number",
    min: "1",
    step: "1",
    suffix: "months",
  },
  {
    name: "Credit_History",
    label: "Credit history",
    type: "select",
    options: [
      { value: "1", label: "Meets credit guidelines" },
      { value: "0", label: "Does not meet guidelines" },
    ],
  },
  {
    name: "Property_Area",
    label: "Property location",
    type: "select",
    options: ["Rural", "Semiurban", "Urban"],
  },
];

// Form controls keep numbers as strings; convert the four numeric values to
// JSON numbers before sending the request to the FastAPI/Pydantic schema.
export function toApplicantPayload(formValues) {
  return {
    ...formValues,
    ApplicantIncome: Number(formValues.ApplicantIncome),
    CoapplicantIncome: Number(formValues.CoapplicantIncome),
    LoanAmount: Number(formValues.LoanAmount),
    Loan_Amount_Term: Number(formValues.Loan_Amount_Term),
    Credit_History: Number(formValues.Credit_History),
  };
}
