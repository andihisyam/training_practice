from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw"
DOCS_DIR = ROOT_DIR / "docs"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
OUTPUTS_DIR = ROOT_DIR / "outputs"

APP_OUTPUT_DIR = OUTPUTS_DIR / "app"
PREPARE_OUTPUT_DIR = APP_OUTPUT_DIR / "prepare_data"
EDA_OUTPUT_DIR = APP_OUTPUT_DIR / "eda"
TRAIN_OUTPUT_DIR = APP_OUTPUT_DIR / "train"

APP_FINAL_DATASET_PATH = PREPARE_OUTPUT_DIR / "final_dataset.csv"
APP_TRANSCRIPT_PATH = PREPARE_OUTPUT_DIR / "transcript_rebuilt.csv"
APP_PREPARE_REPORT_JSON = PREPARE_OUTPUT_DIR / "prepare_report.json"
APP_PREPARE_REPORT_MD = PREPARE_OUTPUT_DIR / "prepare_report.md"

LEGACY_NOTEBOOK_FINAL_DATASET_PATH = OUTPUTS_DIR / "stage_01_03" / "final_dataset_for_eda.csv"
LEGACY_NOTEBOOK_BASELINE_RESULTS_PATH = OUTPUTS_DIR / "stage_01_03" / "baseline_model_results.csv"

RAW_D1_PATH = DATA_DIR / "d1.csv"
RAW_D2_PATH = DATA_DIR / "d2.csv"
RAW_D3_PATH = DATA_DIR / "d3.csv"
RAW_D4_PATH = DATA_DIR / "d4.csv"
RAW_D5_PATH = DATA_DIR / "d5.csv"
RAW_PATIENT_PATH = DATA_DIR / "patient.csv"
RAW_DIAGNOSIS_PATH = DATA_DIR / "diagnosis.csv"
RAW_MEDICATION_PATH = DATA_DIR / "medication.csv"
RAW_PHYSICIAN_PATH = DATA_DIR / "physician_specialty.csv"
RAW_TRANSCRIPT_PATH = DATA_DIR / "transcript.csv"

