"""Opt-in semantic smoke tests against a running local vLLM server."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from text_extraction.structured import VLLMStructuredExtractor, load_schema

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_VLLM_INTEGRATION") != "1",
    reason="set RUN_VLLM_INTEGRATION=1 with a local vLLM server running",
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = os.getenv("VLLM_MODEL", "openai/gpt-oss-20b")
BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")


def extractor(schema_name: str) -> VLLMStructuredExtractor:
    return VLLMStructuredExtractor(
        model=MODEL,
        schema=load_schema(ROOT / "schemas" / schema_name),
        base_url=BASE_URL,
    )


def test_discharge_renal_timing_and_ckd_stage():
    result = extractor("discharge_renal.json").extract(
        "ADMISSION DIAGNOSIS: Pneumonia. HOSPITAL COURSE: On hospital day 3, "
        "the patient developed acute kidney injury after contrast exposure. "
        "Past medical history includes CKD stage 3b."
    )
    assert result["aki_status"] == "hospital_acquired"
    assert result["ckd_stage"] == "3b"


def test_nephrology_does_not_infer_aki_from_esrd_labs():
    result = extractor("nephrology_renal.json").extract(
        "Nephrology consult for dialysis management. ESRD due to diabetic "
        "nephropathy. Hemodialysis Monday, Wednesday, and Friday. Prior failed "
        "renal transplant. Current creatinine 6.8 mg/dL and eGFR 8. Continue HD."
    )
    assert result["reason_for_consult"] == "dialysis_management"
    assert result["aki"] == "absent"
    assert result["aki_stage"] == "not_documented"
    assert result["ckd_stage"] == "5"
    assert result["dialysis_status"] == "hemodialysis"
    assert result["baseline_creatinine"] is None
