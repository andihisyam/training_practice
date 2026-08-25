#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _pct(n: float) -> str:
    if math.isnan(n):
        return "nan"
    return f"{100.0 * n:.2f}%"


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")


def _read_csv(path: Path):
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "This script requires pandas. Install it first, e.g.:\n"
            "  python -m pip install pandas\n"
            f"Original error: {exc}"
        )
    return pd.read_csv(path)


def _optional_import_sklearn():
    try:
        import sklearn  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _optional_import_plotting():
    try:
        import matplotlib  # type: ignore  # noqa: F401
        import matplotlib.pyplot as plt  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


@dataclass(frozen=True)
class FileInfo:
    name: str
    path: Path


def _file_infos(data_dir: Path) -> dict[str, FileInfo]:
    files = {
        "patient": FileInfo("patient.csv", data_dir / "patient.csv"),
        "diagnosis": FileInfo("diagnosis.csv", data_dir / "diagnosis.csv"),
        "medication": FileInfo("medication.csv", data_dir / "medication.csv"),
        "physician_specialty": FileInfo("physician_specialty.csv", data_dir / "physician_specialty.csv"),
        "transcript": FileInfo("transcript.csv", data_dir / "transcript.csv"),
    }
    missing = [v.path for v in files.values() if not v.path.exists()]
    if missing:
        joined = "\n".join(f"- {p}" for p in missing)
        raise SystemExit(f"Missing required CSV file(s):\n{joined}")
    return files


def _basic_table_profile(df, key: str = "PatientGuid") -> dict[str, Any]:
    n_rows = int(df.shape[0])
    n_cols = int(df.shape[1])
    out: dict[str, Any] = {"rows": n_rows, "cols": n_cols}
    if key in df.columns:
        out["key"] = key
        out["key_unique"] = int(df[key].nunique(dropna=True))
        out["key_missing"] = int(df[key].isna().sum())
    out["columns"] = [str(c) for c in df.columns.tolist()]
    return out


def _series_value_counts_dict(s, max_items: int = 20) -> dict[str, int]:
    vc = s.value_counts(dropna=False).head(max_items)
    return {str(k): int(v) for k, v in vc.items()}


def _ensure_dm_indicator(patient_df) -> None:
    if "DMIndicator" not in patient_df.columns:
        raise SystemExit("patient.csv must contain DMIndicator column")
    unique_vals = sorted(set(map(str, patient_df["DMIndicator"].dropna().unique().tolist())))
    if unique_vals != ["0", "1"]:
        # allow numeric 0/1 but show what we saw
        raise SystemExit(f"Unexpected DMIndicator values in patient.csv: {unique_vals}")


def _align_on_patient_guid(base_df, other_df, other_name: str):
    if "PatientGuid" not in other_df.columns:
        raise SystemExit(f"{other_name} is missing PatientGuid column")
    dup = int(other_df["PatientGuid"].duplicated().sum())
    if dup:
        raise SystemExit(f"{other_name} has duplicated PatientGuid rows: {dup} (expected 1 row per patient)")
    return base_df.merge(other_df, on="PatientGuid", how="left", suffixes=("", f"__{other_name}"))


def _feature_columns(df, exclude: Iterable[str]) -> list[str]:
    exclude_set = set(exclude)
    return [c for c in df.columns if c not in exclude_set]


def _top_feature_diffs(df, y_col: str, feature_cols: list[str], top_n: int = 30) -> list[dict[str, Any]]:
    # For count-like columns and normalized meds, mean difference is an interpretable heuristic.
    import pandas as pd  # type: ignore

    dm1 = df[df[y_col] == 1]
    dm0 = df[df[y_col] == 0]
    rows: list[dict[str, Any]] = []

    # Avoid huge memory by processing in chunks of columns.
    chunk = 256
    for start in range(0, len(feature_cols), chunk):
        cols = feature_cols[start : start + chunk]
        # Coerce to numeric; non-numeric becomes NaN and will be ignored.
        x1 = dm1[cols].apply(pd.to_numeric, errors="coerce")
        x0 = dm0[cols].apply(pd.to_numeric, errors="coerce")
        mean1 = x1.mean(axis=0, skipna=True)
        mean0 = x0.mean(axis=0, skipna=True)
        diff = (mean1 - mean0).abs()
        for c in cols:
            rows.append(
                {
                    "feature": c,
                    "mean_dm1": float(mean1.get(c, float("nan"))),
                    "mean_dm0": float(mean0.get(c, float("nan"))),
                    "abs_mean_diff": float(diff.get(c, float("nan"))),
                }
            )
    rows.sort(key=lambda r: (math.isnan(r["abs_mean_diff"]), -r["abs_mean_diff"]))
    return rows[:top_n]


def _fit_simple_models(df, y_col: str, feature_groups: dict[str, list[str]], out_dir: Path) -> dict[str, Any]:
    # Keep this optional so the script still works without sklearn.
    try:
        import numpy as np  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore
        from sklearn.model_selection import StratifiedKFold  # type: ignore
    except Exception:
        return {"skipped": True, "reason": "scikit-learn not installed"}

    y = df[y_col].astype(int).to_numpy()
    results: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for group_name, cols in feature_groups.items():
        if not cols:
            continue

        X = df[cols].apply(lambda c: c.astype("float32", errors="ignore")).to_numpy(dtype=np.float32, copy=False)
        aucs: list[float] = []
        aps: list[float] = []

        for tr, te in cv.split(X, y):
            model = LogisticRegression(
                solver="saga",
                penalty="l2",
                C=1.0,
                max_iter=2000,
                random_state=42,
            )
            model.fit(X[tr], y[tr])
            proba = model.predict_proba(X[te])[:, 1]
            aucs.append(float(roc_auc_score(y[te], proba)))
            aps.append(float(average_precision_score(y[te], proba)))

        results.append(
            {
                "group": group_name,
                "n_features": int(X.shape[1]),
                "roc_auc_mean": float(np.mean(aucs)),
                "roc_auc_std": float(np.std(aucs)),
                "pr_auc_mean": float(np.mean(aps)),
                "pr_auc_std": float(np.std(aps)),
            }
        )

        # Fit on all data once to extract top coefficients.
        final_model = LogisticRegression(
            solver="saga",
            penalty="l2",
            C=1.0,
            max_iter=4000,
            random_state=42,
        )
        final_model.fit(X, y)
        coefs = final_model.coef_.ravel()
        order = np.argsort(np.abs(coefs))[::-1][:40]
        for idx in order:
            coef_rows.append({"group": group_name, "feature": cols[int(idx)], "coef": float(coefs[int(idx)])})

    _write_json(out_dir / "model_metrics.json", {"generated_at": _utc_now_iso(), "metrics": results})
    if coef_rows:
        _write_json(out_dir / "top_coefficients.json", {"generated_at": _utc_now_iso(), "rows": coef_rows})
    return {"skipped": False, "metrics": results}


def _pick_plot_features(joined_df) -> dict[str, list[str]]:
    """
    Keep plots readable and fast:
    - Use a curated set of numeric features (mostly summary stats).
    - Add a few top-diff features from diagnosis/specialty if available.
    """
    import pandas as pd  # type: ignore

    numeric_cols = [c for c in joined_df.columns if c not in {"PatientGuid"}]
    numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(joined_df[c])]

    base = [
        "Age",
        "DiagnosisCount",
        "VisitCount",
        "DiagnosisFreq",
        "AcuteCount",
        "AcuteFreq",
        "BMI_Mean",
        "Weight_Mean",
        "SystolicBP_Mean",
        "DiastolicBP_Mean",
        "Temperature_Mean",
        "RespiratoryRate_Mean",
    ]
    base = [c for c in base if c in numeric_cols]

    # Add a few diagnosis group columns if present
    diag_groups = [c for c in numeric_cols if c.startswith("Icd9_")]
    diag_groups = sorted(diag_groups)[:10]

    # Add a few specialty columns if present
    specs = [c for c in numeric_cols if c.startswith("PhySp_")]
    specs = sorted(specs)[:10]

    corr_cols = []
    seen = set()
    for c in [*base, *diag_groups, *specs]:
        if c in numeric_cols and c not in seen:
            seen.add(c)
            corr_cols.append(c)

    # For boxplots we prefer continuous-ish summary metrics.
    box_cols = [c for c in base if c != "Age"]  # Age gets its own histogram; still fine in corr.

    return {"corr": corr_cols, "box": box_cols, "hist": base}


def _make_plots(joined_df, out_dir: Path) -> dict[str, Any]:
    if not _optional_import_plotting():
        return {"skipped": True, "reason": "matplotlib not installed"}

    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    plots_dir = out_dir / "plots"
    _safe_mkdir(plots_dir)

    # Prepare data
    if "DMIndicator" not in joined_df.columns:
        return {"skipped": True, "reason": "DMIndicator missing from joined_df"}

    df = joined_df.copy()
    df["DMIndicator"] = df["DMIndicator"].astype(int)

    plot_features = _pick_plot_features(df)

    artifacts: dict[str, str] = {}

    # 1) Label distribution
    fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
    vc = df["DMIndicator"].value_counts().sort_index()
    ax.bar([str(i) for i in vc.index.tolist()], vc.values.tolist())
    ax.set_title("DMIndicator Distribution")
    ax.set_xlabel("DMIndicator")
    ax.set_ylabel("Count")
    label_path = plots_dir / "label_distribution.png"
    fig.tight_layout()
    fig.savefig(label_path)
    plt.close(fig)
    artifacts["label_distribution"] = str(label_path.as_posix())

    # 2) A few histograms (DM=0 vs DM=1)
    hist_cols = [c for c in plot_features["hist"] if c in df.columns and c != "DMIndicator"]
    if hist_cols:
        rows = int(math.ceil(len(hist_cols) / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(10, max(4, rows * 3)), dpi=140)
        axes = np.array(axes).reshape(-1)
        for i, col in enumerate(hist_cols):
            ax = axes[i]
            x0 = pd.to_numeric(df.loc[df["DMIndicator"] == 0, col], errors="coerce").dropna()
            x1 = pd.to_numeric(df.loc[df["DMIndicator"] == 1, col], errors="coerce").dropna()
            ax.hist(x0, bins=30, alpha=0.6, label="DM=0", density=True)
            ax.hist(x1, bins=30, alpha=0.6, label="DM=1", density=True)
            ax.set_title(f"Distribution: {col}")
            ax.legend(fontsize=8)
        for j in range(len(hist_cols), len(axes)):
            axes[j].axis("off")
        hist_path = plots_dir / "distributions_selected.png"
        fig.tight_layout()
        fig.savefig(hist_path)
        plt.close(fig)
        artifacts["distributions_selected"] = str(hist_path.as_posix())

    # 3) Boxplots for outlier inspection (by DM group)
    box_cols = [c for c in plot_features["box"] if c in df.columns and c != "DMIndicator"]
    if box_cols:
        rows = int(math.ceil(len(box_cols) / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(10, max(4, rows * 3)), dpi=140)
        axes = np.array(axes).reshape(-1)
        for i, col in enumerate(box_cols):
            ax = axes[i]
            data = [
                pd.to_numeric(df.loc[df["DMIndicator"] == 0, col], errors="coerce").dropna(),
                pd.to_numeric(df.loc[df["DMIndicator"] == 1, col], errors="coerce").dropna(),
            ]
            ax.boxplot(data, tick_labels=["DM=0", "DM=1"], showfliers=True)
            ax.set_title(f"Boxplot (Outliers): {col}")
        for j in range(len(box_cols), len(axes)):
            axes[j].axis("off")
        box_path = plots_dir / "boxplots_selected.png"
        fig.tight_layout()
        fig.savefig(box_path)
        plt.close(fig)
        artifacts["boxplots_selected"] = str(box_path.as_posix())

    # 4) Correlation matrix (selected numeric columns)
    corr_cols = [c for c in plot_features["corr"] if c in df.columns and c != "DMIndicator"]
    if corr_cols:
        corr_df = df[corr_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        corr = corr_df.corr(method="pearson")
        fig, ax = plt.subplots(figsize=(max(8, len(corr_cols) * 0.35), max(6, len(corr_cols) * 0.35)), dpi=140)
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title("Correlation Matrix (Selected Features)")
        ax.set_xticks(range(len(corr_cols)))
        ax.set_yticks(range(len(corr_cols)))
        ax.set_xticklabels(corr_cols, rotation=90, fontsize=7)
        ax.set_yticklabels(corr_cols, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        corr_path = plots_dir / "correlation_matrix.png"
        fig.tight_layout()
        fig.savefig(corr_path)
        plt.close(fig)
        artifacts["correlation_matrix"] = str(corr_path.as_posix())

    _write_json(plots_dir / "plot_features.json", plot_features)
    return {"skipped": False, "artifacts": artifacts, "plots_dir": str(plots_dir.as_posix())}


def _infer_unit_from_median(series, kind: str) -> dict[str, Any]:
    """
    Infer units using coarse heuristics from typical adult distributions.
    This is only used to make "implausible value" checks more informative.
    """
    import pandas as pd  # type: ignore

    s = pd.to_numeric(series, errors="coerce").dropna()
    # Some datasets encode missing as 0; avoid letting zeros dominate median-based unit inference.
    s = s[s > 0]
    if s.empty:
        return {"kind": kind, "unit": "unknown", "median": None}
    median = float(s.median())

    if kind == "height":
        # inches typically ~60-70; cm ~160-180
        unit = "cm" if median > 100 else "in"
        return {"kind": kind, "unit": unit, "median": median}
    if kind == "weight":
        # kg typically ~60-90; lbs ~140-220
        unit = "lb" if median > 130 else "kg"
        return {"kind": kind, "unit": unit, "median": median}
    if kind == "temperature":
        # Fahrenheit ~97-99; Celsius ~36-37
        unit = "F" if median > 60 else "C"
        return {"kind": kind, "unit": unit, "median": median}

    return {"kind": kind, "unit": "unknown", "median": median}


def _collect_violations(df, mask, patient_guid_col: str = "PatientGuid", max_examples: int = 10) -> dict[str, Any]:
    import pandas as pd  # type: ignore

    if isinstance(mask, pd.Series):
        bad = df.loc[mask]
    else:
        bad = df.loc[mask]
    count = int(bad.shape[0])
    examples = []
    if count and patient_guid_col in bad.columns:
        examples = bad[patient_guid_col].astype(str).head(max_examples).tolist()
    return {"count": count, "examples": examples}


def _sanity_checks(joined_df) -> dict[str, Any]:
    """
    Flag values that are impossible (e.g., <=0 where not allowed) or clearly implausible.
    We do not drop/fix data here; we only report counts + example PatientGuid for investigation.
    """
    import pandas as pd  # type: ignore

    df = joined_df.copy()
    checks: dict[str, Any] = {"notes": []}

    # Unit inference (best-effort)
    unit_height = _infer_unit_from_median(df["Height_Mean"], "height") if "Height_Mean" in df.columns else None
    unit_weight = _infer_unit_from_median(df["Weight_Mean"], "weight") if "Weight_Mean" in df.columns else None
    unit_temp = _infer_unit_from_median(df["Temperature_Mean"], "temperature") if "Temperature_Mean" in df.columns else None
    checks["inferred_units"] = {"height": unit_height, "weight": unit_weight, "temperature": unit_temp}

    # Age sanity
    if "Age" in df.columns:
        age = pd.to_numeric(df["Age"], errors="coerce")
        checks["age_le_0"] = _collect_violations(df, age <= 0)
        checks["age_gt_120"] = _collect_violations(df, age > 120)

    # Helper to check min/mean/max ordering
    def _order_checks(prefix: str) -> None:
        mx = f"{prefix}_Max"
        mn = f"{prefix}_Min"
        mean = f"{prefix}_Mean"
        std = f"{prefix}_Std"
        if mn in df.columns and mx in df.columns:
            mn_s = pd.to_numeric(df[mn], errors="coerce")
            mx_s = pd.to_numeric(df[mx], errors="coerce")
            checks[f"{prefix}_min_gt_max"] = _collect_violations(df, (mn_s.notna()) & (mx_s.notna()) & (mn_s > mx_s))
        if mn in df.columns and mean in df.columns:
            mn_s = pd.to_numeric(df[mn], errors="coerce")
            mean_s = pd.to_numeric(df[mean], errors="coerce")
            checks[f"{prefix}_mean_lt_min"] = _collect_violations(
                df, (mn_s.notna()) & (mean_s.notna()) & (mean_s < mn_s)
            )
        if mx in df.columns and mean in df.columns:
            mx_s = pd.to_numeric(df[mx], errors="coerce")
            mean_s = pd.to_numeric(df[mean], errors="coerce")
            checks[f"{prefix}_mean_gt_max"] = _collect_violations(
                df, (mx_s.notna()) & (mean_s.notna()) & (mean_s > mx_s)
            )
        if std in df.columns:
            std_s = pd.to_numeric(df[std], errors="coerce")
            checks[f"{prefix}_std_lt_0"] = _collect_violations(df, std_s < 0)

    for prefix in ["Height", "Weight", "BMI", "SystolicBP", "DiastolicBP", "RespiratoryRate", "Temperature"]:
        _order_checks(prefix)

    # Plausibility checks (broad ranges; only to catch obvious errors like zeros or unit problems)
    def _range_check(col: str, lo: float | None, hi: float | None, name: str) -> None:
        if col not in df.columns:
            return
        s = pd.to_numeric(df[col], errors="coerce")
        if lo is not None:
            checks[f"{name}_lt_{lo}"] = _collect_violations(df, s < lo)
        if hi is not None:
            checks[f"{name}_gt_{hi}"] = _collect_violations(df, s > hi)

    # Blood pressure: impossible or absurd
    _range_check("SystolicBP_Min", 1, None, "sbp_min")  # <=0 impossible
    _range_check("DiastolicBP_Min", 1, None, "dbp_min")
    _range_check("SystolicBP_Mean", 50, 300, "sbp_mean_plausible")
    _range_check("DiastolicBP_Mean", 30, 200, "dbp_mean_plausible")
    if "SystolicBP_Mean" in df.columns and "DiastolicBP_Mean" in df.columns:
        sbp = pd.to_numeric(df["SystolicBP_Mean"], errors="coerce")
        dbp = pd.to_numeric(df["DiastolicBP_Mean"], errors="coerce")
        checks["dbp_gt_sbp_mean"] = _collect_violations(df, (sbp.notna()) & (dbp.notna()) & (dbp > sbp))

    # Respiratory rate: impossible/implausible
    _range_check("RespiratoryRate_Min", 1, None, "rr_min")
    _range_check("RespiratoryRate_Mean", 4, 80, "rr_mean_plausible")

    # BMI
    if "BMI_Min" in df.columns:
        bmi_min = pd.to_numeric(df["BMI_Min"], errors="coerce")
        checks["bmi_min_eq_0"] = _collect_violations(df, bmi_min == 0)
        checks["bmi_min_lt_1_nonzero"] = _collect_violations(df, (bmi_min > 0) & (bmi_min < 1))
    if "BMI_Mean" in df.columns:
        bmi_mean = pd.to_numeric(df["BMI_Mean"], errors="coerce")
        checks["bmi_mean_eq_0"] = _collect_violations(df, bmi_mean == 0)
    _range_check("BMI_Mean", 10, 80, "bmi_mean_plausible")

    # Temperature: infer unit to set a plausible range for mean
    if "Temperature_Mean" in df.columns:
        temp_mean = pd.to_numeric(df["Temperature_Mean"], errors="coerce")
        checks["temp_mean_le_0"] = _collect_violations(df, temp_mean <= 0)
        unit = (unit_temp or {}).get("unit", "unknown")
        if unit == "F":
            _range_check("Temperature_Mean", 90, 110, "temp_mean_plausible_F")
        elif unit == "C":
            _range_check("Temperature_Mean", 32, 43, "temp_mean_plausible_C")
        else:
            checks["notes"].append("Temperature unit could not be inferred; only checked for <=0.")

    # Height/Weight: only check for <=0 as impossible, plus unit-informed plausibility
    if "Height_Mean" in df.columns:
        h = pd.to_numeric(df["Height_Mean"], errors="coerce")
        checks["height_mean_le_0"] = _collect_violations(df, h <= 0)
        unit = (unit_height or {}).get("unit", "unknown")
        if unit == "in":
            _range_check("Height_Mean", 48, 84, "height_mean_plausible_in")
        elif unit == "cm":
            _range_check("Height_Mean", 120, 220, "height_mean_plausible_cm")
    if "Weight_Mean" in df.columns:
        w = pd.to_numeric(df["Weight_Mean"], errors="coerce")
        checks["weight_mean_le_0"] = _collect_violations(df, w <= 0)
        unit = (unit_weight or {}).get("unit", "unknown")
        if unit == "lb":
            _range_check("Weight_Mean", 66, 660, "weight_mean_plausible_lb")
        elif unit == "kg":
            _range_check("Weight_Mean", 30, 300, "weight_mean_plausible_kg")

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="EDA + sanity checks for PracticeFusion diabetes prediction dataset (5 CSV files)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Folder containing the CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/eda"),
        help="Output folder for reports/artifacts.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode (default). Kept for compatibility; use --full to include slow steps.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include slow steps (read medication.csv and optionally baseline model).",
    )
    parser.add_argument(
        "--baseline-model",
        action="store_true",
        help="Run baseline logistic regression (requires scikit-learn). Can be slow with high-dimensional features.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top features to output for each group (by abs mean difference).",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir
    _safe_mkdir(out_dir)

    files = _file_infos(data_dir)
    patient_df = _read_csv(files["patient"].path)
    _ensure_dm_indicator(patient_df)

    # Coerce DMIndicator to int for consistent downstream ops.
    patient_df["DMIndicator"] = patient_df["DMIndicator"].astype(int)

    diagnosis_df = _read_csv(files["diagnosis"].path)
    physician_df = _read_csv(files["physician_specialty"].path)
    transcript_df = _read_csv(files["transcript"].path)

    medication_df = None
    run_full = bool(args.full)
    if run_full:
        medication_df = _read_csv(files["medication"].path)
        if "DMIndicator" in medication_df.columns:
            medication_df["DMIndicator"] = medication_df["DMIndicator"].astype(int)

    profile: dict[str, Any] = {
        "generated_at": _utc_now_iso(),
        "data_dir": str(data_dir.resolve()),
        "files": {k: {"path": str(v.path.resolve()), "bytes": v.path.stat().st_size} for k, v in files.items()},
        "tables": {},
        "checks": {},
    }

    # Basic profiles
    profile["tables"]["patient"] = _basic_table_profile(patient_df)
    profile["tables"]["diagnosis"] = _basic_table_profile(diagnosis_df)
    profile["tables"]["physician_specialty"] = _basic_table_profile(physician_df)
    profile["tables"]["transcript"] = _basic_table_profile(transcript_df)
    if medication_df is not None:
        profile["tables"]["medication"] = _basic_table_profile(medication_df)

    # Key-set sanity checks
    patient_keys = set(patient_df["PatientGuid"].astype(str).tolist())
    profile["checks"]["patient_keys"] = len(patient_keys)
    for name, df in [
        ("diagnosis", diagnosis_df),
        ("physician_specialty", physician_df),
        ("transcript", transcript_df),
    ]:
        keys = set(df["PatientGuid"].astype(str).tolist())
        profile["checks"][f"{name}_keys"] = len(keys)
        profile["checks"][f"{name}_missing_from_patient"] = len(keys - patient_keys)
        profile["checks"][f"patient_missing_from_{name}"] = len(patient_keys - keys)

    if medication_df is not None:
        med_keys = set(medication_df["PatientGuid"].astype(str).tolist())
        profile["checks"]["medication_keys"] = len(med_keys)
        profile["checks"]["medication_missing_from_patient"] = len(med_keys - patient_keys)
        profile["checks"]["patient_missing_from_medication"] = len(patient_keys - med_keys)

        if "DMIndicator" in medication_df.columns:
            # Check label consistency for overlapping keys.
            joined = patient_df[["PatientGuid", "DMIndicator"]].merge(
                medication_df[["PatientGuid", "DMIndicator"]], on="PatientGuid", how="inner", suffixes=("_pat", "_med")
            )
            mismatch = int((joined["DMIndicator_pat"] != joined["DMIndicator_med"]).sum())
            profile["checks"]["dm_indicator_mismatch_patient_vs_medication"] = mismatch

    # Demographic distributions
    profile["patient_demographics"] = {
        "dm_indicator_counts": _series_value_counts_dict(patient_df["DMIndicator"]),
        "age_min": int(patient_df["Age"].min()),
        "age_max": int(patient_df["Age"].max()),
        "gender_counts": _series_value_counts_dict(patient_df["Gender"]),
        "state_counts_top20": _series_value_counts_dict(patient_df["State"], max_items=20),
    }

    # Build a joined modeling frame (left join on patient)
    joined_df = patient_df.copy()
    joined_df = _align_on_patient_guid(joined_df, diagnosis_df, "diagnosis")
    joined_df = _align_on_patient_guid(joined_df, physician_df, "physician_specialty")
    joined_df = _align_on_patient_guid(joined_df, transcript_df, "transcript")
    if medication_df is not None:
        # Keep patient label as canonical. Drop med label from features.
        drop_cols = ["DMIndicator"] if "DMIndicator" in medication_df.columns else []
        med_features = medication_df.drop(columns=drop_cols)
        joined_df = _align_on_patient_guid(joined_df, med_features, "medication")

    # Missingness report (NaN-based; zeros are not treated as missing)
    missing_cols = joined_df.isna().mean(axis=0).sort_values(ascending=False)
    top_missing = missing_cols.head(30)
    profile["joined_missing_top30"] = {str(k): float(v) for k, v in top_missing.items()}

    # Sanity checks for impossible / implausible values
    sanity = _sanity_checks(joined_df)
    profile["sanity_checks"] = sanity

    # Feature diffs per group (heuristic)
    top_n = int(args.top_n)
    diffs: dict[str, Any] = {}
    diffs["diagnosis"] = _top_feature_diffs(
        joined_df,
        y_col="DMIndicator",
        feature_cols=_feature_columns(diagnosis_df, exclude=["PatientGuid"]),
        top_n=top_n,
    )
    diffs["physician_specialty"] = _top_feature_diffs(
        joined_df,
        y_col="DMIndicator",
        feature_cols=_feature_columns(physician_df, exclude=["PatientGuid"]),
        top_n=top_n,
    )
    diffs["transcript"] = _top_feature_diffs(
        joined_df,
        y_col="DMIndicator",
        feature_cols=_feature_columns(transcript_df, exclude=["PatientGuid"]),
        top_n=top_n,
    )
    if medication_df is not None:
        diffs["medication"] = _top_feature_diffs(
            joined_df,
            y_col="DMIndicator",
            feature_cols=_feature_columns(medication_df, exclude=["PatientGuid", "DMIndicator"]),
            top_n=top_n,
        )
    profile["top_feature_diffs"] = diffs

    # Plots (run in both fast/full; medication features are not plotted by default)
    plots = _make_plots(joined_df, out_dir=out_dir)
    profile["plots"] = plots

    # Optional modeling
    models: dict[str, Any] = {"skipped": True, "reason": "fast mode"}
    if run_full and args.baseline_model and _optional_import_sklearn():
        # Fill NaNs with 0 for modeling (common for this dataset; NaN usually means missing source row)
        model_df = joined_df.fillna(0)
        feature_groups = {
            "patient": _feature_columns(patient_df, exclude=["PatientGuid", "DMIndicator"]),
            "diagnosis": _feature_columns(diagnosis_df, exclude=["PatientGuid"]),
            "physician_specialty": _feature_columns(physician_df, exclude=["PatientGuid"]),
            "transcript": _feature_columns(transcript_df, exclude=["PatientGuid"]),
            "medication": _feature_columns(medication_df, exclude=["PatientGuid", "DMIndicator"]) if medication_df is not None else [],
        }
        feature_groups["all"] = [
            *feature_groups["patient"],
            *feature_groups["diagnosis"],
            *feature_groups["physician_specialty"],
            *feature_groups["transcript"],
            *feature_groups["medication"],
        ]
        models = _fit_simple_models(model_df, y_col="DMIndicator", feature_groups=feature_groups, out_dir=out_dir)
    elif run_full and args.baseline_model and (not _optional_import_sklearn()):
        models = {"skipped": True, "reason": "scikit-learn not installed"}
    profile["models"] = models

    # Write artifacts
    _write_json(out_dir / "profile.json", profile)

    # Human-readable report
    dm_counts = profile["patient_demographics"]["dm_indicator_counts"]
    dm0 = dm_counts.get("0", 0)
    dm1 = dm_counts.get("1", 0)
    total = dm0 + dm1
    dm_rate = (dm1 / total) if total else float("nan")

    report_lines: list[str] = []
    report_lines.append("# PracticeFusion EDA Report (DM Prediction)")
    report_lines.append("")
    report_lines.append(f"- Generated: `{profile['generated_at']}`")
    report_lines.append(f"- Data dir: `{profile['data_dir']}`")
    report_lines.append("")
    report_lines.append("## Label (DMIndicator)")
    report_lines.append(f"- DM=1: `{dm1}`")
    report_lines.append(f"- DM=0: `{dm0}`")
    report_lines.append(f"- Prevalence: `{_pct(dm_rate)}`")
    report_lines.append("")
    report_lines.append("## Key Observations")
    report_lines.append("- All tables are 1-row-per-patient keyed by `PatientGuid`.")
    if medication_df is not None:
        report_lines.append(
            f"- `medication.csv` is missing for `{profile['checks']['patient_missing_from_medication']}` patient(s); recommend left-join then fill NaN with 0 for med features."
        )
        mismatch = profile["checks"].get("dm_indicator_mismatch_patient_vs_medication", 0)
        report_lines.append(f"- `DMIndicator` matches between `patient.csv` and `medication.csv` for overlapping patients (mismatches: `{mismatch}`).")
    report_lines.append("- `diagnosis.csv` includes derived ratios: `DiagnosisFreq = DiagnosisCount/VisitCount` and `AcuteFreq = AcuteCount/DiagnosisCount`.")
    report_lines.append("")
    report_lines.append("## Potential Leakage Risk")
    report_lines.append(
        "- If `DMIndicator` was constructed from diagnoses/medications, models using those same features can look unrealistically strong; validate with a time-based split if possible."
    )
    report_lines.append("")

    report_lines.append("## Sanity Checks (Impossible / Implausible)")
    report_lines.append(
        "- These checks are for data quality (e.g., zeros, unit mistakes). They do not imply rows should be dropped automatically."
    )
    inf = sanity.get("inferred_units", {})
    if inf:
        h_u = (inf.get("height") or {}).get("unit")
        w_u = (inf.get("weight") or {}).get("unit")
        t_u = (inf.get("temperature") or {}).get("unit")
        report_lines.append(f"- Inferred units (heuristic): height=`{h_u}`, weight=`{w_u}`, temperature=`{t_u}`.")
    # Show a small set of high-signal checks
    key_checks = [
        ("age_le_0", "Age <= 0"),
        ("age_gt_120", "Age > 120"),
        ("sbp_min_lt_1", "SystolicBP_Min < 1 (impossible)"),
        ("dbp_min_lt_1", "DiastolicBP_Min < 1 (impossible)"),
        ("dbp_gt_sbp_mean", "DiastolicBP_Mean > SystolicBP_Mean"),
        ("temp_mean_le_0", "Temperature_Mean <= 0 (impossible)"),
        ("rr_min_lt_1", "RespiratoryRate_Min < 1 (impossible)"),
        ("height_mean_le_0", "Height_Mean <= 0 (impossible)"),
        ("weight_mean_le_0", "Weight_Mean <= 0 (impossible)"),
        ("bmi_min_eq_0", "BMI_Min = 0 (often missing encoded as 0)"),
        ("bmi_min_lt_1_nonzero", "BMI_Min in (0,1) (implausible)"),
        ("bmi_mean_eq_0", "BMI_Mean = 0 (often missing encoded as 0)"),
        ("SystolicBP_min_gt_max", "SystolicBP_Min > SystolicBP_Max"),
        ("Temperature_min_gt_max", "Temperature_Min > Temperature_Max"),
    ]
    for k, label in key_checks:
        item = sanity.get(k)
        if isinstance(item, dict) and "count" in item:
            count = item.get("count", 0)
            examples = (item.get("examples", []) or [])[:5]
            ex_txt = f" examples={examples}" if examples else ""
            report_lines.append(f"- {label}: `{count}`{ex_txt}")
    notes = sanity.get("notes") or []
    for n in notes[:5]:
        report_lines.append(f"- Note: {n}")
    report_lines.append("")

    report_lines.append("## Plots")
    if plots.get("skipped"):
        report_lines.append(f"- Skipped: `{plots.get('reason', 'unknown')}`")
        report_lines.append("- Install matplotlib to enable plots: `python -m pip install matplotlib`")
    else:
        arts = plots.get("artifacts", {})
        report_lines.append(f"- Label distribution: `{arts.get('label_distribution', '')}`")
        if arts.get("distributions_selected"):
            report_lines.append(f"- Selected feature distributions: `{arts.get('distributions_selected')}`")
        if arts.get("boxplots_selected"):
            report_lines.append(f"- Boxplots (outliers): `{arts.get('boxplots_selected')}`")
        if arts.get("correlation_matrix"):
            report_lines.append(f"- Correlation matrix: `{arts.get('correlation_matrix')}`")
        report_lines.append(f"- Plot feature list: `{(out_dir / 'plots' / 'plot_features.json').as_posix()}`")
    report_lines.append("")

    def _append_top(name: str, rows: list[dict[str, Any]]) -> None:
        report_lines.append(f"## Top Feature Differences: {name}")
        report_lines.append("(sorted by absolute mean difference between DM=1 vs DM=0)")
        report_lines.append("")
        for r in rows[: min(15, len(rows))]:
            report_lines.append(
                f"- `{r['feature']}`: mean(DM=1)={r['mean_dm1']:.4g}, mean(DM=0)={r['mean_dm0']:.4g}, |diff|={r['abs_mean_diff']:.4g}"
            )
        report_lines.append("")

    _append_top("diagnosis", diffs["diagnosis"])
    _append_top("physician_specialty", diffs["physician_specialty"])
    _append_top("transcript", diffs["transcript"])
    if medication_df is not None and "medication" in diffs:
        _append_top("medication", diffs["medication"])

    if models.get("skipped") is False:
        report_lines.append("## Baseline Models (Logistic Regression, 5-fold CV)")
        report_lines.append("")
        for m in models.get("metrics", []):
            report_lines.append(
                f"- `{m['group']}`: features={m['n_features']}, ROC-AUC={m['roc_auc_mean']:.4f}±{m['roc_auc_std']:.4f}, PR-AUC={m['pr_auc_mean']:.4f}±{m['pr_auc_std']:.4f}"
            )
        report_lines.append("")
        report_lines.append("Artifacts:")
        report_lines.append(f"- `{(out_dir / 'model_metrics.json').as_posix()}`")
        if (out_dir / "top_coefficients.json").exists():
            report_lines.append(f"- `{(out_dir / 'top_coefficients.json').as_posix()}`")
        report_lines.append("")
    else:
        report_lines.append("## Baseline Models")
        report_lines.append(f"- Skipped: `{models.get('reason', 'not requested')}`")
        report_lines.append("- Run with `--full --baseline-model` to enable (requires scikit-learn).")
        report_lines.append("")

    report_lines.append("## Artifacts")
    report_lines.append(f"- `{(out_dir / 'profile.json').as_posix()}`")
    report_lines.append(f"- `{(out_dir / 'report.md').as_posix()}`")
    report_lines.append("")

    _write_text(out_dir / "report.md", "\n".join(report_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
