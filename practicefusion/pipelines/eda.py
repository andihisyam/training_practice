from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from practicefusion.config import APP_FINAL_DATASET_PATH, EDA_OUTPUT_DIR
from practicefusion.utils.io import ensure_dir, load_csv, save_json, save_markdown


def _safe_import_seaborn():
    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid")
        return sns
    except Exception:
        return None


def top_mean_diff(df: pd.DataFrame, columns: list[str], top_n: int = 10) -> pd.DataFrame:
    group_mean = df.groupby("DMIndicator")[columns].mean().T
    group_mean.columns = ["mean_non_dm", "mean_dm"]
    group_mean["abs_diff"] = (group_mean["mean_dm"] - group_mean["mean_non_dm"]).abs()
    return group_mean.sort_values("abs_diff", ascending=False).head(top_n)


def run_eda(dataset_path: Path = APP_FINAL_DATASET_PATH, out_dir: Path = EDA_OUTPUT_DIR) -> dict[str, Any]:
    ensure_dir(out_dir)
    plots_dir = ensure_dir(out_dir / "plots")
    sns = _safe_import_seaborn()

    df = load_csv(dataset_path)
    diagnosis_cols = [c for c in df.columns if c.startswith("Icd9_") or c in ["DiagnosisCount", "VisitCount", "DiagnosisFreq", "AcuteCount", "AcuteFreq"]]
    physician_cols = [c for c in df.columns if c.startswith("PhySp_")]

    label_counts = df["DMIndicator"].value_counts().sort_index()
    label_pct = (label_counts / label_counts.sum() * 100).round(2)
    label_summary = pd.DataFrame({"count": label_counts, "pct": label_pct})

    quality_checks = pd.DataFrame(
        [
            {"metric": "Jumlah pasien final", "count": int(df.shape[0])},
            {"metric": "Jumlah fitur final", "count": int(df.shape[1] - 2)},
        ]
    )

    selected_numeric = [
        "Age",
        "DiagnosisCount",
        "VisitCount",
        "BMI_Mean",
        "Weight_Mean",
        "SystolicBP_Mean",
        "DiastolicBP_Mean",
        "RespiratoryRate_Mean",
        "Temperature_Mean",
        "Height_NObs",
        "Weight_NObs",
    ]
    selected_numeric = [c for c in selected_numeric if c in df.columns]

    summary_by_target = df.groupby("DMIndicator")[selected_numeric].mean().T
    summary_by_target.columns = ["mean_non_dm", "mean_dm"]
    summary_by_target["abs_diff"] = (summary_by_target["mean_dm"] - summary_by_target["mean_non_dm"]).abs()
    summary_by_target = summary_by_target.sort_values("abs_diff", ascending=False)

    diagnosis_top = top_mean_diff(df, [c for c in diagnosis_cols if c in df.columns], top_n=10)
    physician_top = top_mean_diff(df, [c for c in physician_cols if c in df.columns], top_n=10)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Non-DM (0)", "DM (1)"], label_counts.values, color=["#94A3B8", "#1D4ED8"])
    ax.set_title("Distribusi DMIndicator")
    ax.set_ylabel("Jumlah pasien")
    for i, v in enumerate(label_counts.values):
        ax.text(i, v + max(label_counts.values) * 0.01, f"{v:,}", ha="center")
    fig.tight_layout()
    fig.savefig(plots_dir / "label_distribution.png")
    plt.close(fig)

    if selected_numeric:
        n_cols = 2
        n_rows = int(np.ceil(len(selected_numeric) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = np.array(axes).reshape(-1)
        for idx, col in enumerate(selected_numeric):
            ax = axes[idx]
            if sns is not None:
                sns.kdeplot(data=df, x=col, hue="DMIndicator", fill=True, common_norm=False, ax=ax)
            else:
                ax.hist(df.loc[df["DMIndicator"] == 0, col].dropna(), bins=30, alpha=0.55, label="0")
                ax.hist(df.loc[df["DMIndicator"] == 1, col].dropna(), bins=30, alpha=0.55, label="1")
                ax.legend(title="DMIndicator")
            ax.set_title(f"Distribusi {col}")
        for idx in range(len(selected_numeric), len(axes)):
            axes[idx].axis("off")
        fig.tight_layout()
        fig.savefig(plots_dir / "selected_numeric_distribution.png")
        plt.close(fig)

    boxplot_cols = [c for c in ["BMI_Mean", "Weight_Mean", "SystolicBP_Mean", "DiastolicBP_Mean"] if c in df.columns]
    if boxplot_cols:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = np.array(axes).reshape(-1)
        for idx, col in enumerate(boxplot_cols):
            ax = axes[idx]
            if sns is not None:
                sns.boxplot(data=df, x="DMIndicator", y=col, ax=ax)
            else:
                df.boxplot(column=col, by="DMIndicator", ax=ax)
            ax.set_title(f"Boxplot {col}")
            ax.set_xlabel("DMIndicator")
        for idx in range(len(boxplot_cols), len(axes)):
            axes[idx].axis("off")
        fig.tight_layout()
        fig.savefig(plots_dir / "selected_boxplots.png")
        plt.close(fig)

    corr_cols = [c for c in selected_numeric if c in df.columns]
    if corr_cols:
        corr_df = df[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        if sns is not None:
            sns.heatmap(corr_df, annot=True, cmap="Blues", fmt=".2f", ax=ax)
        else:
            im = ax.imshow(corr_df, cmap="Blues")
            ax.set_xticks(range(len(corr_cols)))
            ax.set_xticklabels(corr_cols, rotation=90)
            ax.set_yticks(range(len(corr_cols)))
            ax.set_yticklabels(corr_cols)
            fig.colorbar(im, ax=ax)
        ax.set_title("Korelasi fitur numerik terpilih")
        fig.tight_layout()
        fig.savefig(plots_dir / "selected_correlation.png")
        plt.close(fig)

    report_payload = {
        "dataset_path": str(dataset_path),
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "label_summary": label_summary.reset_index().rename(columns={"index": "DMIndicator"}).to_dict(orient="records"),
        "quality_checks": quality_checks.to_dict(orient="records"),
        "selected_numeric_summary": summary_by_target.reset_index().rename(columns={"index": "feature"}).to_dict(orient="records"),
        "top_diagnosis_diff": diagnosis_top.reset_index().rename(columns={"index": "feature"}).to_dict(orient="records"),
        "top_physician_diff": physician_top.reset_index().rename(columns={"index": "feature"}).to_dict(orient="records"),
    }
    save_json(out_dir / "eda_summary.json", report_payload)

    lines: list[str] = []
    lines.append("# EDA Report")
    lines.append("")
    lines.append(f"- Dataset: `{dataset_path}`")
    lines.append(f"- Shape: `{df.shape[0]} x {df.shape[1]}`")
    lines.append("")
    lines.append("## Distribusi Label")
    lines.append("")
    for row in report_payload["label_summary"]:
        lines.append(f"- `DMIndicator={row['DMIndicator']}`: count=`{row['count']}`, pct=`{row['pct']}`")
    lines.append("")
    lines.append("## Quality Checks")
    lines.append("")
    for row in report_payload["quality_checks"]:
        lines.append(f"- `{row['metric']}`: `{row['count']}`")
    lines.append("")
    lines.append("## Top Diagnosis Mean Differences")
    lines.append("")
    for row in report_payload["top_diagnosis_diff"][:10]:
        lines.append(f"- `{row['feature']}`: |diff|=`{row['abs_diff']:.4f}`")
    lines.append("")
    lines.append("## Top Physician Mean Differences")
    lines.append("")
    for row in report_payload["top_physician_diff"][:10]:
        lines.append(f"- `{row['feature']}`: |diff|=`{row['abs_diff']:.4f}`")
    lines.append("")
    lines.append("## Kesimpulan Singkat")
    lines.append("")
    lines.append("- Distribusi label masih tidak seimbang, sehingga recall, F1-score, dan PR-AUC penting pada tahap modelling.")
    lines.append("- Fitur transcript hasil rebuild tetap memberi sinyal klinis yang berguna, terutama pada BMI, berat badan, dan tekanan darah.")
    lines.append("- Diagnosis, physician specialty, dan medication tetap perlu diawasi karena dapat mendekati label diabetes terlalu kuat.")
    save_markdown(out_dir / "report.md", "\n".join(lines))

    return {
        "dataset_path": dataset_path,
        "out_dir": out_dir,
        "summary_path": out_dir / "eda_summary.json",
        "report_path": out_dir / "report.md",
        "plots_dir": plots_dir,
    }

