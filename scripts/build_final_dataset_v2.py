from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REBUILT_TRANSCRIPT_PATH = ROOT / "outputs" / "rebuild_transcript_from_d2" / "transcript_rebuilt_from_d2.csv"
OUT_DIR = ROOT / "outputs" / "final_dataset_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_CSV_PATH = OUT_DIR / "final_dataset_v2.csv"
FINAL_REPORT_JSON = OUT_DIR / "final_dataset_v2_report.json"
FINAL_REPORT_MD = OUT_DIR / "final_dataset_v2_report.md"


def load_data() -> dict[str, pd.DataFrame]:
    d5 = pd.read_csv(RAW_DIR / "d5.csv")
    old_patient = pd.read_csv(RAW_DIR / "patient.csv")
    diagnosis = pd.read_csv(RAW_DIR / "diagnosis.csv")
    medication = pd.read_csv(RAW_DIR / "medication.csv")
    physician = pd.read_csv(RAW_DIR / "physician_specialty.csv")
    transcript = pd.read_csv(REBUILT_TRANSCRIPT_PATH)
    return {
        "d5": d5,
        "old_patient": old_patient,
        "diagnosis": diagnosis,
        "medication": medication,
        "physician": physician,
        "transcript": transcript,
    }


def build_patient_backbone(d5: pd.DataFrame, old_patient: pd.DataFrame) -> pd.DataFrame:
    old_age = old_patient[["PatientGuid", "Age"]].copy()
    backbone = d5.rename(columns={"dmIndicator": "DMIndicator"}).copy()
    backbone["DMIndicator"] = backbone["DMIndicator"].astype(int)
    backbone = backbone.merge(old_age, on="PatientGuid", how="left")

    # Put age beside YearOfBirth for readability.
    ordered = [
        "PatientGuid",
        "DMIndicator",
        "Gender",
        "YearOfBirth",
        "Age",
        "State",
        "PracticeGuid",
    ]
    return backbone[ordered]


def write_reports(final_df: pd.DataFrame, report: dict[str, Any]) -> None:
    FINAL_REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Final Dataset V2 Report")
    lines.append("")
    lines.append(f"- Output file: `{FINAL_CSV_PATH}`")
    lines.append(f"- Final rows: `{report['final_shape']['rows']}`")
    lines.append(f"- Final cols: `{report['final_shape']['cols']}`")
    lines.append("")
    lines.append("## Cohort Rules")
    lines.append("")
    for note in report["cohort_rules"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Join Summary")
    lines.append("")
    for row in report["join_summary"]:
        lines.append(
            f"- `{row['table']}`: source_rows=`{row['source_rows']}`, source_patients=`{row['source_patients']}`, "
            f"missing_patients_from_base=`{row['missing_patients_from_base']}`"
        )
    lines.append("")
    lines.append("## Patients Dropped")
    lines.append("")
    for row in report["dropped_summary"]:
        lines.append(f"- `{row['reason']}`: `{row['n_patients']}`")
    lines.append("")
    lines.append("## Final Sample")
    lines.append("")
    lines.append("```csv")
    lines.append(final_df.head(5).to_csv(index=False).strip())
    lines.append("```")
    FINAL_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    data = load_data()
    d5 = data["d5"]
    diagnosis = data["diagnosis"]
    medication = data["medication"]
    physician = data["physician"]
    transcript = data["transcript"]

    backbone = build_patient_backbone(data["d5"], data["old_patient"])
    base_keys = set(backbone["PatientGuid"])

    join_tables = {
        "diagnosis": diagnosis,
        "medication": medication,
        "physician_specialty": physician,
        "transcript_rebuilt": transcript,
    }

    join_summary: list[dict[str, Any]] = []
    common_keys = set(base_keys)
    for name, df in join_tables.items():
        keys = set(df["PatientGuid"])
        common_keys &= keys
        join_summary.append(
            {
                "table": name,
                "source_rows": int(df.shape[0]),
                "source_patients": int(df["PatientGuid"].nunique()),
                "missing_patients_from_base": int(len(base_keys - keys)),
            }
        )

    dropped_summary = [
        {"reason": "No medication record", "n_patients": int(len(base_keys - set(medication["PatientGuid"])))},
        {"reason": "No rebuilt transcript record", "n_patients": int(len(base_keys - set(transcript["PatientGuid"])))},
        {"reason": "Dropped by final intersection of all required tables", "n_patients": int(len(base_keys - common_keys))},
    ]

    final_backbone = backbone[backbone["PatientGuid"].isin(common_keys)].copy()

    medication_features = medication.drop(columns=["DMIndicator"], errors="ignore")
    final_df = (
        final_backbone
        .merge(diagnosis, on="PatientGuid", how="inner")
        .merge(medication_features, on="PatientGuid", how="inner")
        .merge(physician, on="PatientGuid", how="inner")
        .merge(transcript, on="PatientGuid", how="inner")
    )

    # Remove accidental duplicate columns if any remain after source cleanup.
    final_df = final_df.loc[:, ~final_df.columns.duplicated()].copy()
    final_df = final_df.drop(columns=[c for c in final_df.columns if c.startswith("HeartRate_")], errors="ignore")

    # Keep patient backbone columns first, followed by other feature blocks.
    patient_cols = ["PatientGuid", "DMIndicator", "Gender", "YearOfBirth", "Age", "State", "PracticeGuid"]
    remaining_cols = [c for c in final_df.columns if c not in patient_cols]
    final_df = final_df[patient_cols + remaining_cols]

    report = {
        "final_shape": {"rows": int(final_df.shape[0]), "cols": int(final_df.shape[1])},
        "cohort_rules": [
            "Patient/label backbone comes from d5.csv.",
            "Gender is taken from d5.csv in raw form (M/F).",
            "Age is retained from the older patient.csv because d5.csv does not carry age directly.",
            "Diagnosis, medication, and physician specialty features are kept from the old aggregated dataset.",
            "Transcript features are fully rebuilt from d2.csv.",
            "Final cohort is the inner intersection of d5, diagnosis, medication, physician_specialty, and rebuilt transcript.",
        ],
        "join_summary": join_summary,
        "dropped_summary": dropped_summary,
    }

    final_df.to_csv(FINAL_CSV_PATH, index=False)
    write_reports(final_df, report)

    print(f"Saved final dataset to: {FINAL_CSV_PATH}")
    print(f"Saved report to: {FINAL_REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
