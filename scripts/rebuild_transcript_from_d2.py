from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs" / "rebuild_transcript_from_d2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

D2_PATH = DATA_DIR / "d2.csv"
OLD_TRANSCRIPT_PATH = DATA_DIR / "transcript.csv"

REBUILT_CSV_PATH = OUT_DIR / "transcript_rebuilt_from_d2.csv"
REPORT_JSON_PATH = OUT_DIR / "transcript_rebuild_report.json"
REPORT_MD_PATH = OUT_DIR / "transcript_rebuild_report.md"


@dataclass(frozen=True)
class VariableRule:
    raw_col: str
    out_prefix: str
    low: float | None = None
    high: float | None = None


VARIABLE_RULES: list[VariableRule] = [
    VariableRule("Height", "Height", 48, 84),
    VariableRule("Weight", "Weight", 50, 700),
    VariableRule("BMI", "BMI", 10, 80),
    VariableRule("SystolicBP", "SystolicBP", 50, 300),
    VariableRule("DiastolicBP", "DiastolicBP", 30, 200),
    VariableRule("RespiratoryRate", "RespiratoryRate", 4, 80),
    VariableRule("Temperature", "Temperature", 90, 110),
]


def clean_series(series: pd.Series, low: float | None, high: float | None) -> pd.Series:
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


def rebuild_transcript(d2_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = d2_df.copy()
    working["VisitYear"] = pd.to_numeric(working["VisitYear"], errors="coerce")

    cleaning_summary: list[dict[str, Any]] = []

    for rule in VARIABLE_RULES:
        raw = pd.to_numeric(working[rule.raw_col], errors="coerce")
        cleaned = clean_series(working[rule.raw_col], rule.low, rule.high)
        working[rule.raw_col] = cleaned

        cleaning_summary.append(
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

    patient_index = pd.DataFrame({"PatientGuid": sorted(working["PatientGuid"].dropna().unique())})
    rebuilt = patient_index.copy()

    for rule in VARIABLE_RULES:
        agg = (
            working.groupby("PatientGuid")[rule.raw_col]
            .agg(["min", "max", "mean", "std", "count"])
            .rename(
                columns={
                    "min": f"{rule.out_prefix}_Min",
                    "max": f"{rule.out_prefix}_Max",
                    "mean": f"{rule.out_prefix}_Mean",
                    "std": f"{rule.out_prefix}_Std",
                    "count": f"{rule.out_prefix}_NObs",
                }
            )
            .reset_index()
        )
        agg[f"{rule.out_prefix}_NObs"] = agg[f"{rule.out_prefix}_NObs"].astype(int)

        change = build_change_by_year(working, rule.raw_col).reset_index()
        change = change.rename(columns={f"{rule.raw_col}_Change": f"{rule.out_prefix}_Change"})

        rebuilt = rebuilt.merge(agg, on="PatientGuid", how="left")
        rebuilt = rebuilt.merge(change, on="PatientGuid", how="left")

    # Match the old transcript column order first, then append the new fields.
    old_transcript_cols = pd.read_csv(OLD_TRANSCRIPT_PATH, nrows=0).columns.tolist()
    base_cols = [col for col in old_transcript_cols if col in rebuilt.columns]
    extra_cols = [col for col in rebuilt.columns if col not in base_cols and col != "PatientGuid"]
    rebuilt = rebuilt[["PatientGuid", *[c for c in base_cols if c != "PatientGuid"], *extra_cols]]

    summary = {
        "source_rows": int(d2_df.shape[0]),
        "source_patients": int(d2_df["PatientGuid"].nunique()),
        "rebuilt_rows": int(rebuilt.shape[0]),
        "rebuilt_cols": int(rebuilt.shape[1]),
        "cleaning_summary": cleaning_summary,
        "notes": [
            "Transcript rebuilt entirely from d2.csv.",
            "Absolute vitals were cleaned with value-range rules before aggregation.",
            "Change is defined as mean(latest VisitYear) - mean(earliest VisitYear) per patient.",
            "NObs columns count valid non-missing observations after cleaning.",
        ],
    }
    return rebuilt, summary


def write_report(rebuilt_df: pd.DataFrame, summary: dict[str, Any]) -> None:
    REPORT_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Transcript Rebuild Report")
    lines.append("")
    lines.append(f"- Source file: `{D2_PATH}`")
    lines.append(f"- Output file: `{REBUILT_CSV_PATH}`")
    lines.append(f"- Source rows: `{summary['source_rows']}`")
    lines.append(f"- Source patients: `{summary['source_patients']}`")
    lines.append(f"- Rebuilt rows: `{summary['rebuilt_rows']}`")
    lines.append(f"- Rebuilt cols: `{summary['rebuilt_cols']}`")
    lines.append("")
    lines.append("## Cleaning Summary")
    lines.append("")
    for row in summary["cleaning_summary"]:
        lines.append(
            f"- `{row['variable']}`: raw_missing=`{row['raw_missing']}`, raw_zero=`{row['raw_zero']}`, "
            f"clean_missing=`{row['clean_missing']}`, invalidated_by_rules=`{row['invalidated_by_rules']}`, "
            f"range=`[{row['low_threshold']}, {row['high_threshold']}]`"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in summary["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Sample")
    lines.append("")
    sample_csv = rebuilt_df.head(5).to_csv(index=False).strip()
    lines.append("```csv")
    lines.append(sample_csv)
    lines.append("```")
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    d2_df = pd.read_csv(D2_PATH)
    rebuilt_df, summary = rebuild_transcript(d2_df)
    rebuilt_df.to_csv(REBUILT_CSV_PATH, index=False)
    write_report(rebuilt_df, summary)

    print(f"Saved rebuilt transcript to: {REBUILT_CSV_PATH}")
    print(f"Saved report to: {REPORT_MD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
