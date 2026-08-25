from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from practicefusion.config import (
    APP_FINAL_DATASET_PATH,
    APP_PREPARE_REPORT_JSON,
    APP_PREPARE_REPORT_MD,
    APP_TRANSCRIPT_PATH,
    PREPARE_OUTPUT_DIR,
    RAW_D2_PATH,
    RAW_D5_PATH,
    RAW_DIAGNOSIS_PATH,
    RAW_MEDICATION_PATH,
    RAW_PATIENT_PATH,
    RAW_PHYSICIAN_PATH,
)
from practicefusion.utils.io import ensure_dir, load_csv, save_csv, save_json, save_markdown


@dataclass(frozen=True)
class VariableRule:
    raw_col: str
    out_prefix: str
    low: float | None = None
    high: float | None = None


TRANSCRIPT_RULES: list[VariableRule] = [
    VariableRule("Height", "Height", 48, 84),
    VariableRule("Weight", "Weight", 50, 700),
    VariableRule("BMI", "BMI", 10, 80),
    VariableRule("SystolicBP", "SystolicBP", 50, 300),
    VariableRule("DiastolicBP", "DiastolicBP", 30, 200),
    VariableRule("RespiratoryRate", "RespiratoryRate", 4, 80),
    VariableRule("Temperature", "Temperature", 90, 110),
]
STD_SUFFIX = "_Std"
MISSING_DROP_THRESHOLD = 0.40
TRANSCRIPT_VALUE_SUFFIXES = ("_Min", "_Max", "_Mean")


def clean_transcript_series(series: pd.Series, low: float | None, high: float | None) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce")
    cleaned = cleaned.mask(cleaned <= 0)
    if low is not None:
        cleaned = cleaned.mask(cleaned < low)
    if high is not None:
        cleaned = cleaned.mask(cleaned > high)
    return cleaned


def build_change_by_year(df: pd.DataFrame, value_col: str) -> pd.Series:
    valid = df.loc[df[value_col].notna(), ["PatientGuid", "VisitYear", value_col]].copy()
    if valid.empty:
        return pd.Series(
            data=[],
            index=pd.Index([], name="PatientGuid"),
            dtype=float,
            name=f"{value_col}_Change",
        )

    yearly = (
        valid.groupby(["PatientGuid", "VisitYear"], as_index=False)[value_col]
        .mean()
        .sort_values(["PatientGuid", "VisitYear"])
    )
    change = yearly.groupby("PatientGuid")[value_col].agg(lambda s: float(s.iloc[-1] - s.iloc[0]))
    change.name = f"{value_col}_Change"
    return change


def build_missing_indicator_name(column_name: str) -> str:
    return f"IsImputed_{column_name}"


def apply_final_missing_value_strategy(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    prepared = df.copy()

    std_cols = [c for c in prepared.columns if c.endswith(STD_SUFFIX)]
    prepared = prepared.drop(columns=std_cols, errors="ignore")

    candidate_feature_cols = [
        c
        for c in prepared.columns
        if c not in {"PatientGuid", "DMIndicator"}
    ]
    missing_rates = prepared[candidate_feature_cols].isna().mean()
    high_missing_cols = missing_rates[missing_rates > MISSING_DROP_THRESHOLD].index.tolist()
    prepared = prepared.drop(columns=high_missing_cols, errors="ignore")

    feature_cols_after_drop = [
        c
        for c in prepared.columns
        if c not in {"PatientGuid", "DMIndicator"}
    ]
    missing_flag_cols: list[str] = []
    flag_summary_rows: list[dict[str, Any]] = []
    for col in feature_cols_after_drop:
        missing_count = int(prepared[col].isna().sum())
        if missing_count <= 0:
            continue
        flag_col = build_missing_indicator_name(col)
        prepared[flag_col] = prepared[col].isna().astype(int)
        missing_flag_cols.append(flag_col)
        flag_summary_rows.append(
            {
                "column": col,
                "missing_count": missing_count,
                "missing_rate": float(prepared[col].isna().mean()),
                "flag_column": flag_col,
            }
        )

    strategy_report = {
        "std_cols_dropped": std_cols,
        "missing_drop_threshold": MISSING_DROP_THRESHOLD,
        "high_missing_cols_dropped": high_missing_cols,
        "missing_flag_cols_added": missing_flag_cols,
        "missing_flag_summary": flag_summary_rows,
    }
    return prepared, strategy_report


def load_prepare_sources() -> dict[str, pd.DataFrame]:
    return {
        "d2": load_csv(RAW_D2_PATH),
        "d5": load_csv(RAW_D5_PATH),
        "patient_old": load_csv(RAW_PATIENT_PATH),
        "diagnosis_old": load_csv(RAW_DIAGNOSIS_PATH),
        "medication_old": load_csv(RAW_MEDICATION_PATH),
        "physician_old": load_csv(RAW_PHYSICIAN_PATH),
    }


def rebuild_transcript_from_d2(d2_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = d2_df.copy()
    working["VisitYear"] = pd.to_numeric(working["VisitYear"], errors="coerce")

    cleaning_log_rows: list[dict[str, Any]] = []
    for rule in TRANSCRIPT_RULES:
        raw = pd.to_numeric(working[rule.raw_col], errors="coerce")
        cleaned = clean_transcript_series(working[rule.raw_col], low=rule.low, high=rule.high)
        working[rule.raw_col] = cleaned
        cleaning_log_rows.append(
            {
                "variable": rule.raw_col,
                "raw_missing": int(raw.isna().sum()),
                "raw_zero": int((raw == 0).sum()),
                "clean_missing": int(cleaned.isna().sum()),
                "invalidated_by_rules": int(cleaned.isna().sum() - raw.isna().sum()),
                "low_threshold": rule.low,
                "high_threshold": rule.high,
            }
        )

    transcript = pd.DataFrame({"PatientGuid": sorted(working["PatientGuid"].dropna().unique())})
    for rule in TRANSCRIPT_RULES:
        agg = (
            working.groupby("PatientGuid")[rule.raw_col]
            .agg(["min", "max", "mean", "count"])
            .rename(
                columns={
                    "min": f"{rule.out_prefix}_Min",
                    "max": f"{rule.out_prefix}_Max",
                    "mean": f"{rule.out_prefix}_Mean",
                    "count": f"{rule.out_prefix}_NObs",
                }
            )
            .reset_index()
        )
        agg[f"{rule.out_prefix}_NObs"] = agg[f"{rule.out_prefix}_NObs"].astype(int)

        change = build_change_by_year(working, rule.raw_col).reset_index()
        change = change.rename(columns={f"{rule.raw_col}_Change": f"{rule.out_prefix}_Change"})

        transcript = transcript.merge(agg, on="PatientGuid", how="left")
        transcript = transcript.merge(change, on="PatientGuid", how="left")

    cleaning_log_df = pd.DataFrame(cleaning_log_rows)
    return transcript, cleaning_log_df


def build_final_dataset(sources: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    d5 = sources["d5"]
    patient_old = sources["patient_old"]
    diagnosis_old = sources["diagnosis_old"]
    medication_old = sources["medication_old"]
    physician_old = sources["physician_old"]
    transcript_rebuilt = sources["transcript_rebuilt"]

    patient_backbone = d5.rename(columns={"dmIndicator": "DMIndicator"}).copy()
    patient_backbone["DMIndicator"] = patient_backbone["DMIndicator"].astype(int)
    patient_backbone = patient_backbone.merge(patient_old[["PatientGuid", "Age"]], on="PatientGuid", how="left")
    patient_backbone = patient_backbone[
        ["PatientGuid", "DMIndicator", "Gender", "YearOfBirth", "Age", "State", "PracticeGuid"]
    ]

    medication_features_final = medication_old.drop(columns=["DMIndicator"], errors="ignore")
    base_keys = set(patient_backbone["PatientGuid"])
    transcript_keys = set(transcript_rebuilt["PatientGuid"])
    required_keys = base_keys & transcript_keys

    patient_backbone_final = patient_backbone[patient_backbone["PatientGuid"].isin(required_keys)].copy()
    final_dataset = patient_backbone_final.merge(transcript_rebuilt, on="PatientGuid", how="inner")

    optional_sources = {
        "diagnosis": diagnosis_old,
        "physician_specialty": physician_old,
        "medication": medication_features_final,
    }
    join_summary: list[dict[str, Any]] = [
        {
            "table": "transcript_rebuilt",
            "join_type": "required_inner",
            "source_rows": int(transcript_rebuilt.shape[0]),
            "source_patients": int(transcript_rebuilt["PatientGuid"].nunique()),
            "missing_patients_from_backbone": int(len(base_keys - transcript_keys)),
            "coverage_in_final_cohort": round(float(len(required_keys) / len(base_keys)), 4) if base_keys else 0.0,
        }
    ]

    for name, source_df in optional_sources.items():
        source_keys = set(source_df["PatientGuid"])
        source_feature_cols = [col for col in source_df.columns if col != "PatientGuid"]
        final_dataset = final_dataset.merge(source_df, on="PatientGuid", how="left")
        if source_feature_cols:
            final_dataset[source_feature_cols] = final_dataset[source_feature_cols].fillna(0)
        join_summary.append(
            {
                "table": name,
                "join_type": "optional_left",
                "source_rows": int(source_df.shape[0]),
                "source_patients": int(source_df["PatientGuid"].nunique()),
                "missing_patients_from_backbone": int(len(base_keys - source_keys)),
                "coverage_in_final_cohort": round(
                    float(len(required_keys & source_keys) / len(required_keys)),
                    4,
                )
                if required_keys
                else 0.0,
            }
        )

    dropped_summary = [
        {
            "reason": "No rebuilt transcript record",
            "n_patients": int(len(base_keys - transcript_keys)),
        },
        {
            "reason": "Dropped by required backbone + transcript intersection",
            "n_patients": int(len(base_keys - required_keys)),
        },
    ]

    final_dataset = final_dataset.loc[:, ~final_dataset.columns.duplicated()].copy()
    final_dataset = final_dataset.drop(columns=[c for c in final_dataset.columns if c.startswith("HeartRate_")], errors="ignore")
    final_dataset, missing_strategy_report = apply_final_missing_value_strategy(final_dataset)

    patient_cols = ["PatientGuid", "DMIndicator", "Gender", "YearOfBirth", "Age", "State", "PracticeGuid"]
    remaining_cols = [c for c in final_dataset.columns if c not in patient_cols]
    final_dataset = final_dataset[patient_cols + remaining_cols]

    report = {
        "final_shape": {"rows": int(final_dataset.shape[0]), "cols": int(final_dataset.shape[1])},
        "join_summary": join_summary,
        "dropped_summary": dropped_summary,
        "notes": [
            "Patient and label backbone comes from d5.csv.",
            "Age is retained from patient.csv because d5.csv does not provide Age directly.",
            "Transcript is rebuilt entirely from d2.csv and becomes the only required clinical table besides the patient backbone.",
            "Diagnosis, physician specialty, and medication are retained as optional left-joined aggregated blocks so the final cohort does not collapse when one block is absent.",
            "Optional aggregated blocks are filled with zero when a patient has no matching row in that source table.",
            "Final cohort uses the inner intersection of patient backbone and rebuilt transcript only.",
            "Transcript std columns are removed from the final dataset.",
            "Columns with missing rate above 40% are dropped.",
            "Explicit missing indicator columns are added to the final dataset for any feature that will be imputed later.",
        ],
        "missing_strategy": missing_strategy_report,
    }
    return final_dataset, report


def write_prepare_report(report: dict[str, Any], cleaning_log_df: pd.DataFrame) -> None:
    ensure_dir(PREPARE_OUTPUT_DIR)
    save_json(APP_PREPARE_REPORT_JSON, report)

    lines: list[str] = []
    lines.append("# Prepare Data Report")
    lines.append("")
    lines.append(f"- Final dataset path: `{APP_FINAL_DATASET_PATH}`")
    lines.append(f"- Transcript rebuild path: `{APP_TRANSCRIPT_PATH}`")
    lines.append(f"- Final shape: `{report['final_shape']['rows']} x {report['final_shape']['cols']}`")
    lines.append("")
    lines.append("## Cohort Notes")
    lines.extend(f"- {note}" for note in report["notes"])
    lines.append("")
    lines.append("## Join Summary")
    lines.append("")
    for row in report["join_summary"]:
        lines.append(
            f"- `{row['table']}` ({row['join_type']}): rows=`{row['source_rows']}`, patients=`{row['source_patients']}`, "
            f"missing_from_backbone=`{row['missing_patients_from_backbone']}`, coverage_in_final_cohort=`{row['coverage_in_final_cohort']:.4f}`"
        )
    lines.append("")
    lines.append("## Dropped Patients")
    lines.append("")
    for row in report["dropped_summary"]:
        lines.append(f"- `{row['reason']}`: `{row['n_patients']}`")
    lines.append("")
    lines.append("## Transcript Cleaning Summary")
    lines.append("")
    for row in cleaning_log_df.to_dict(orient="records"):
        lines.append(
            f"- `{row['variable']}`: raw_missing=`{row['raw_missing']}`, raw_zero=`{row['raw_zero']}`, "
            f"clean_missing=`{row['clean_missing']}`, invalidated=`{row['invalidated_by_rules']}`"
        )
    lines.append("")
    lines.append("## Final Missing Strategy")
    lines.append("")
    lines.append(f"- Missing drop threshold: `{report['missing_strategy']['missing_drop_threshold']:.2f}`")
    lines.append(f"- Dropped std columns: `{len(report['missing_strategy']['std_cols_dropped'])}`")
    if report["missing_strategy"]["std_cols_dropped"]:
        lines.append(f"- Std columns removed: `{', '.join(report['missing_strategy']['std_cols_dropped'])}`")
    lines.append(f"- Dropped high-missing columns: `{len(report['missing_strategy']['high_missing_cols_dropped'])}`")
    if report["missing_strategy"]["high_missing_cols_dropped"]:
        lines.append(f"- High-missing columns removed: `{', '.join(report['missing_strategy']['high_missing_cols_dropped'])}`")
    lines.append(f"- Missing indicator columns added: `{len(report['missing_strategy']['missing_flag_cols_added'])}`")
    for row in report["missing_strategy"]["missing_flag_summary"]:
        lines.append(
            f"- `{row['column']}` -> `{row['flag_column']}`: missing_count=`{row['missing_count']}`, missing_rate=`{row['missing_rate']:.4f}`"
        )

    save_markdown(APP_PREPARE_REPORT_MD, "\n".join(lines))


def run_prepare_data(
    final_dataset_path: Path = APP_FINAL_DATASET_PATH,
    transcript_output_path: Path = APP_TRANSCRIPT_PATH,
) -> dict[str, Any]:
    ensure_dir(PREPARE_OUTPUT_DIR)
    sources = load_prepare_sources()
    transcript_rebuilt, cleaning_log_df = rebuild_transcript_from_d2(sources["d2"])
    sources["transcript_rebuilt"] = transcript_rebuilt

    final_dataset, report = build_final_dataset(sources)
    save_csv(transcript_rebuilt, transcript_output_path)
    save_csv(final_dataset, final_dataset_path)
    write_prepare_report(report, cleaning_log_df)

    return {
        "final_dataset": final_dataset,
        "transcript_rebuilt": transcript_rebuilt,
        "cleaning_log": cleaning_log_df,
        "report": report,
        "final_dataset_path": final_dataset_path,
        "transcript_output_path": transcript_output_path,
    }
