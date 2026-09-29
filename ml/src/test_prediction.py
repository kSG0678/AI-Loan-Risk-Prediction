"""Tests for packaged single-applicant inference and fresh-process loading."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

if __package__:
    from .data_preprocessing import DEFAULT_DATA_PATH
    from .pipeline import MODEL_PATH, load_inference_pipeline
    from .predict import predict_applicant
else:
    from data_preprocessing import DEFAULT_DATA_PATH
    from pipeline import MODEL_PATH, load_inference_pipeline
    from predict import predict_applicant, predict_with_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REALISTIC_APPLICANT = {
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
    "Property_Area": "Semiurban",
}


class PredictionPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Packaged pipeline is missing; run ml/src/pipeline.py first: {MODEL_PATH}"
            )

    def test_saved_pipeline_predicts_one_applicant(self) -> None:
        before_hash = hashlib.sha256(DEFAULT_DATA_PATH.read_bytes()).hexdigest()
        result = predict_applicant(REALISTIC_APPLICANT)

        self.assertIn(result["predicted_class"], {"Y", "N"})
        self.assertEqual(
            result["predicted_class_label"],
            {"Y": "Approved", "N": "Rejected"}[result["predicted_class"]],
        )
        self.assertGreaterEqual(result["approval_probability"], 0.0)
        self.assertLessEqual(result["approval_probability"], 1.0)
        self.assertGreaterEqual(result["rejection_probability"], 0.0)
        self.assertLessEqual(result["rejection_probability"], 1.0)
        self.assertAlmostEqual(
            result["approval_probability"] + result["rejection_probability"],
            1.0,
        )
        self.assertEqual(
            before_hash, hashlib.sha256(DEFAULT_DATA_PATH.read_bytes()).hexdigest()
        )

    def test_prediction_loads_artifact_without_training(self) -> None:
        with (
            patch(
                "ml.src.pipeline.build_inference_pipeline",
                side_effect=AssertionError("Inference must not rebuild the pipeline."),
            ),
            patch(
                "ml.src.pipeline.prepare_data",
                side_effect=AssertionError("Inference must not prepare training data."),
            ),
        ):
            result = predict_applicant(REALISTIC_APPLICANT)
        self.assertIn(result["predicted_class"], {"Y", "N"})

    def test_pipeline_contains_complete_inference_steps(self) -> None:
        pipeline = load_inference_pipeline()
        self.assertEqual(
            [name for name, _ in pipeline.steps],
            ["feature_engineering", "preprocessor", "classifier"],
        )

    def test_target_identifier_and_missing_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Loan_Status"):
            predict_applicant({**REALISTIC_APPLICANT, "Loan_Status": "Y"})
        with self.assertRaisesRegex(ValueError, "Loan_ID"):
            predict_applicant({**REALISTIC_APPLICANT, "Loan_ID": "LP001003"})
        incomplete = dict(REALISTIC_APPLICANT)
        del incomplete["Property_Area"]
        with self.assertRaisesRegex(ValueError, "Property_Area"):
            predict_applicant(incomplete)

    def test_saved_pipeline_loads_and_predicts_in_fresh_process(self) -> None:
        script = (
            "import json; "
            "from ml.src.predict import predict_applicant; "
            f"record = json.loads({json.dumps(json.dumps(REALISTIC_APPLICANT))}); "
            "print(json.dumps(predict_applicant(record)))"
        )
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(PROJECT_ROOT)
            if not existing_pythonpath
            else str(PROJECT_ROOT) + os.pathsep + existing_pythonpath
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertIn(result["predicted_class"], {"Y", "N"})
        self.assertAlmostEqual(
            result["approval_probability"] + result["rejection_probability"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
