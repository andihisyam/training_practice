from __future__ import annotations

import os
import re
import sys
import webbrowser
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from practicefusion.config import APP_FINAL_DATASET_PATH, EDA_OUTPUT_DIR, PREPARE_OUTPUT_DIR, TRAIN_OUTPUT_DIR
from practicefusion.pipelines.eda import run_eda
from practicefusion.pipelines.prepare_data import run_prepare_data
from practicefusion.pipelines.train import (
    describe_feature_set,
    run_error_analysis,
    run_feature_set_screening,
    run_feature_leakage_audit,
    run_leakage_check,
    run_methodology_checks,
    run_main_cv,
    run_optuna_tuning,
    run_smote_comparison,
    run_threshold_tuning,
    run_training,
    run_weighting_comparison,
    run_xai_analysis,
)

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    Console = None
    Markdown = None
    Panel = None
    Table = None


console = Console() if HAS_RICH else None

APP_TITLE = "PracticeFusion Research Runner"
APP_SUBTITLE = "Klasifikasi Diabetes Mellitus Tipe 2 dengan XAI"

REPORT_PATHS = {
    "Kesimpulan umum": TRAIN_OUTPUT_DIR / "general_conclusion.md",
    "Ringkasan penelitian": TRAIN_OUTPUT_DIR / "rebuild_research_summary.md",
    "Report prepare data": PREPARE_OUTPUT_DIR / "prepare_report.md",
    "Report EDA": EDA_OUTPUT_DIR / "report.md",
    "Report feature set screening": TRAIN_OUTPUT_DIR / "feature_set_screening_report.md",
    "Report methodology check": TRAIN_OUTPUT_DIR / "methodology_checks_report.md",
    "Report training": TRAIN_OUTPUT_DIR / "report.md",
    "Report leakage audit": TRAIN_OUTPUT_DIR / "leakage_audit_report.md",
    "Report leakage check": TRAIN_OUTPUT_DIR / "leakage_check_report.md",
    "Report weighting": TRAIN_OUTPUT_DIR / "weighting_comparison_report.md",
    "Report SMOTE": TRAIN_OUTPUT_DIR / "smote_comparison_report.md",
    "Report cross validation": TRAIN_OUTPUT_DIR / "main_cv_report.md",
    "Report Optuna": TRAIN_OUTPUT_DIR / "optuna_report.md",
    "Report threshold tuning": TRAIN_OUTPUT_DIR / "threshold_tuning_report.md",
    "Report error analysis": TRAIN_OUTPUT_DIR / "error_analysis_report.md",
    "Report XAI": TRAIN_OUTPUT_DIR / "xai" / "xai_report.md",
}

VISUAL_PATHS = {
    "SHAP Summary Plot": TRAIN_OUTPUT_DIR / "xai" / "shap_summary_plot.png",
    "SHAP Summary Bar Plot": TRAIN_OUTPUT_DIR / "xai" / "shap_summary_bar.png",
    "SHAP Dependence Age": TRAIN_OUTPUT_DIR / "xai" / "shap_dependence_Age.png",
    "SHAP Dependence BMI_Max": TRAIN_OUTPUT_DIR / "xai" / "shap_dependence_BMI_Max.png",
    "SHAP Dependence DiastolicBP_Mean": TRAIN_OUTPUT_DIR / "xai" / "shap_dependence_DiastolicBP_Mean.png",
    "Force Plot TP": TRAIN_OUTPUT_DIR / "xai" / "xai_force_tp_high_confidence.html",
    "Force Plot FN": TRAIN_OUTPUT_DIR / "xai" / "xai_force_fn_borderline.html",
    "Force Plot FP": TRAIN_OUTPUT_DIR / "xai" / "xai_force_fp_high_confidence.html",
    "Force Plot TN": TRAIN_OUTPUT_DIR / "xai" / "xai_force_tn_borderline.html",
}


def run_interactive_menu() -> int:
    while True:
        choice = prompt_menu(
            title="Menu Utama",
            options=[
                ("1", "Jalankan Pipeline / Eksperimen"),
                ("2", "Lihat Ringkasan Hasil"),
                ("3", "Buka Visualisasi / Report"),
                ("4", "Kesimpulan Umum"),
                ("0", "Keluar"),
            ],
        )

        if choice == "1":
            run_pipeline_menu()
        elif choice == "2":
            run_results_menu()
        elif choice == "3":
            run_visual_menu()
        elif choice == "4":
            show_general_conclusion()
        elif choice == "0":
            print_message("Sampai jumpa.")
            return 0
        else:
            print_message("Pilihan tidak dikenali. Coba lagi.")


def run_pipeline_menu() -> None:
    while True:
        choice = prompt_menu(
            title="Jalankan Pipeline / Eksperimen",
            options=[
                ("1", "Prepare Data"),
                ("2", "EDA"),
                ("3", "Feature Set Screening"),
                ("4", "Cross Validation Model"),
                ("5", "Methodology Check"),
                ("6", "Leakage Audit"),
                ("7", "Leakage Check"),
                ("8", "Weighting Comparison"),
                ("9", "SMOTE Comparison"),
                ("10", "Optuna Tuning"),
                ("11", "Threshold Tuning"),
                ("12", "Final Evaluation"),
                ("13", "Error Analysis"),
                ("14", "XAI"),
                ("15", "Jalankan Semua Tahap Bersih"),
                ("0", "Kembali"),
            ],
        )

        if choice == "0":
            return

        actions: dict[str, tuple[str, Callable[[], Any]]] = {
            "1": ("Prepare Data", lambda: run_prepare_data(final_dataset_path=APP_FINAL_DATASET_PATH)),
            "2": ("EDA", lambda: run_eda(dataset_path=APP_FINAL_DATASET_PATH, out_dir=EDA_OUTPUT_DIR)),
            "3": ("Feature Set Screening", lambda: run_feature_set_screening(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)),
            "4": ("Cross Validation Model", lambda: run_main_cv(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)),
            "5": ("Methodology Check", lambda: run_methodology_checks(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)),
            "6": ("Leakage Audit", lambda: run_feature_leakage_audit(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)),
            "7": ("Leakage Check", lambda: run_leakage_check(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)),
            "8": ("Weighting Comparison", lambda: run_weighting_comparison(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)),
            "9": ("SMOTE Comparison", lambda: run_smote_comparison(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)),
            "10": ("Optuna Tuning", lambda: run_optuna_tuning(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)),
            "11": ("Threshold Tuning", lambda: run_threshold_tuning(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)),
            "12": ("Final Evaluation", lambda: run_training(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)),
            "13": ("Error Analysis", lambda: run_error_analysis(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)),
            "14": ("XAI", lambda: run_xai_analysis(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)),
            "15": ("Semua Tahap Bersih", run_all_clean_steps),
        }

        label, action = actions.get(choice, (None, None))
        if action is None:
            print_message("Pilihan tidak dikenali. Coba lagi.")
            continue

        run_action(label, action)


def run_results_menu() -> None:
    while True:
        choice = prompt_menu(
            title="Lihat Ringkasan Hasil",
            options=[
                ("1", "Lihat ringkasan penelitian"),
                ("2", "Lihat report prepare data"),
                ("3", "Lihat report EDA"),
                ("4", "Lihat report feature set screening"),
                ("5", "Lihat report methodology check"),
                ("6", "Lihat report cross validation"),
                ("7", "Lihat report Optuna"),
                ("8", "Lihat report threshold tuning"),
                ("9", "Lihat report training"),
                ("10", "Lihat report leakage audit"),
                ("11", "Lihat report leakage check"),
                ("12", "Lihat report weighting"),
                ("13", "Lihat report SMOTE"),
                ("14", "Lihat report error analysis"),
                ("15", "Lihat report XAI"),
                ("16", "Lihat top fitur SHAP"),
                ("17", "Lihat status artefak"),
                ("18", "Lihat kesimpulan umum"),
                ("0", "Kembali"),
            ],
        )

        if choice == "0":
            return

        report_choices = {
            "1": "Ringkasan penelitian",
            "2": "Report prepare data",
            "3": "Report EDA",
            "4": "Report feature set screening",
            "5": "Report methodology check",
            "6": "Report cross validation",
            "7": "Report Optuna",
            "8": "Report threshold tuning",
            "9": "Report training",
            "10": "Report leakage audit",
            "11": "Report leakage check",
            "12": "Report weighting",
            "13": "Report SMOTE",
            "14": "Report error analysis",
            "15": "Report XAI",
        }

        if choice in report_choices:
            show_text_file(REPORT_PATHS[report_choices[choice]])
        elif choice == "16":
            show_top_shap_features()
        elif choice == "17":
            show_artifact_status()
        elif choice == "18":
            show_general_conclusion()
        else:
            print_message("Pilihan tidak dikenali. Coba lagi.")


def run_visual_menu() -> None:
    while True:
        choice = prompt_menu(
            title="Buka Visualisasi / Report",
            options=[
                ("1", "Buka SHAP Summary Plot"),
                ("2", "Buka SHAP Summary Bar Plot"),
                ("3", "Buka SHAP Dependence Age"),
                ("4", "Buka SHAP Dependence BMI_Max"),
                ("5", "Buka SHAP Dependence DiastolicBP_Mean"),
                ("6", "Buka Force Plot TP"),
                ("7", "Buka Force Plot FN"),
                ("8", "Buka Force Plot FP"),
                ("9", "Buka Force Plot TN"),
                ("10", "Buka report XAI"),
                ("0", "Kembali"),
            ],
        )

        if choice == "0":
            return

        visual_choices = {
            "1": VISUAL_PATHS["SHAP Summary Plot"],
            "2": VISUAL_PATHS["SHAP Summary Bar Plot"],
            "3": VISUAL_PATHS["SHAP Dependence Age"],
            "4": VISUAL_PATHS["SHAP Dependence BMI_Max"],
            "5": VISUAL_PATHS["SHAP Dependence DiastolicBP_Mean"],
            "6": VISUAL_PATHS["Force Plot TP"],
            "7": VISUAL_PATHS["Force Plot FN"],
            "8": VISUAL_PATHS["Force Plot FP"],
            "9": VISUAL_PATHS["Force Plot TN"],
            "10": REPORT_PATHS["Report XAI"],
        }

        target_path = visual_choices.get(choice)
        if target_path is None:
            print_message("Pilihan tidak dikenali. Coba lagi.")
            continue
        open_artifact(target_path)


def run_all_clean_steps() -> dict[str, Any]:
    results: dict[str, Any] = {}
    results["prepare_data"] = run_prepare_data(final_dataset_path=APP_FINAL_DATASET_PATH)
    results["eda"] = run_eda(dataset_path=APP_FINAL_DATASET_PATH, out_dir=EDA_OUTPUT_DIR)
    results["methodology_check"] = run_methodology_checks(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)
    results["feature_set_screening"] = run_feature_set_screening(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)
    results["cv"] = run_main_cv(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)
    results["leakage_audit"] = run_feature_leakage_audit(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)
    results["leakage_check"] = run_leakage_check(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)
    results["weighting"] = run_weighting_comparison(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)
    results["smote"] = run_smote_comparison(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)
    results["optuna"] = run_optuna_tuning(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)
    results["threshold"] = run_threshold_tuning(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)
    results["training"] = run_training(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)
    results["error_analysis"] = run_error_analysis(
        dataset_path=APP_FINAL_DATASET_PATH,
        out_dir=TRAIN_OUTPUT_DIR,
    )
    results["xai"] = run_xai_analysis(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)
    return results


def run_action(label: str, action: Callable[[], Any]) -> None:
    print_message(f"Menjalankan: {label}")
    try:
        result = action()
    except Exception as exc:  # noqa: BLE001
        print_message(f"Terjadi error saat menjalankan `{label}`: {exc}")
        pause()
        return

    summarize_action_result(label, result)
    pause()


def summarize_action_result(label: str, result: Any) -> None:
    if isinstance(result, dict):
        key_paths = {
            key: value
            for key, value in result.items()
            if isinstance(value, Path)
        }
        if key_paths:
            lines = [f"{key}: {value}" for key, value in key_paths.items()]
            print_block(f"{label} selesai", "\n".join(lines))
            return
    print_message(f"{label} selesai.")


def show_text_file(path: Path) -> None:
    if not path.exists():
        print_message(f"File belum tersedia: {path}")
        pause()
        return

    content = path.read_text(encoding="utf-8")
    if HAS_RICH and Markdown is not None:
        console.print(Panel(Markdown(content), title=path.name, expand=True))
    else:
        separator = "=" * 72
        print(separator)
        print(path.name)
        print(separator)
        print(content)
    pause()


def show_top_shap_features(limit: int = 10) -> None:
    path = TRAIN_OUTPUT_DIR / "xai" / "xai_top_features.csv"
    if not path.exists():
        print_message("File top fitur SHAP belum tersedia.")
        pause()
        return

    df = pd.read_csv(path).head(limit)
    if HAS_RICH and Table is not None:
        table = Table(title="Top Fitur SHAP", show_lines=False)
        table.add_column("Feature")
        table.add_column("Mean |SHAP|", justify="right")
        for row in df.to_dict(orient="records"):
            table.add_row(str(row["feature"]), f"{float(row['mean_abs_shap']):.4f}")
        console.print(table)
    else:
        print(df[["feature", "mean_abs_shap"]].to_string(index=False))
    pause()


def show_artifact_status() -> None:
    status_items = {
        "Final dataset": APP_FINAL_DATASET_PATH,
        "Prepare report": REPORT_PATHS["Report prepare data"],
        "EDA report": REPORT_PATHS["Report EDA"],
        "Feature set screening report": REPORT_PATHS["Report feature set screening"],
        "Methodology check report": REPORT_PATHS["Report methodology check"],
        "CV report": REPORT_PATHS["Report cross validation"],
        "Optuna report": REPORT_PATHS["Report Optuna"],
        "Threshold report": REPORT_PATHS["Report threshold tuning"],
        "Training report": REPORT_PATHS["Report training"],
        "Leakage audit report": REPORT_PATHS["Report leakage audit"],
        "Leakage check report": TRAIN_OUTPUT_DIR / "leakage_check_report.md",
        "Weighting report": REPORT_PATHS["Report weighting"],
        "SMOTE report": REPORT_PATHS["Report SMOTE"],
        "Error analysis report": REPORT_PATHS["Report error analysis"],
        "XAI report": REPORT_PATHS["Report XAI"],
        "SHAP summary plot": VISUAL_PATHS["SHAP Summary Plot"],
    }

    rows = []
    for label, path in status_items.items():
        exists = path.exists()
        rows.append((label, "[OK]" if exists else "[--]", str(path)))

    if HAS_RICH and Table is not None:
        table = Table(title="Status Artefak")
        table.add_column("Artefak")
        table.add_column("Status", justify="center")
        table.add_column("Path")
        for label, status, path_str in rows:
            table.add_row(label, status, path_str)
        console.print(table)
    else:
        print("Status Artefak")
        print("-" * 72)
        for label, status, path_str in rows:
            print(f"{status} {label}: {path_str}")
    pause()


def show_general_conclusion() -> None:
    content = build_general_conclusion_text()
    REPORT_PATHS["Kesimpulan umum"].write_text(content, encoding="utf-8", newline="\n")

    if HAS_RICH and Markdown is not None:
        console.print(Panel(Markdown(content), title="Kesimpulan Umum", expand=True))
    else:
        print_block("Kesimpulan Umum", content)
    pause()


def build_general_conclusion_text() -> str:
    baseline_path = TRAIN_OUTPUT_DIR / "baseline_results.csv"
    cv_path = TRAIN_OUTPUT_DIR / "main_cv_summary.csv"
    threshold_path = TRAIN_OUTPUT_DIR / "threshold_tuning_best_thresholds.csv"
    shap_path = TRAIN_OUTPUT_DIR / "xai" / "xai_top_features.csv"
    prepare_report_path = PREPARE_OUTPUT_DIR / "prepare_report.md"

    if not baseline_path.exists():
        return "# Kesimpulan Umum\n\nHasil training utama belum tersedia. Jalankan baseline training terlebih dahulu."

    baseline_df = pd.read_csv(baseline_path)
    primary_df = baseline_df[baseline_df["feature_set_role"] == "primary"].copy()
    best_primary = primary_df.iloc[0] if not primary_df.empty else baseline_df.iloc[0]
    strategy = describe_feature_set(str(best_primary["feature_set"]))

    shape_text = get_dataset_shape_text(prepare_report_path)
    cv_text = get_cv_summary_text(cv_path)
    threshold_text = get_threshold_summary_text(threshold_path, str(best_primary["feature_set"]), str(best_primary["model"]))
    xai_text = get_xai_summary_text(shap_path)
    rationale_text = get_feature_set_rationale(str(best_primary["feature_set"]))

    lines: list[str] = []
    lines.append("# Kesimpulan Umum")
    lines.append("")
    lines.append("## Gambaran Singkat")
    lines.append("")
    lines.append(f"- Dataset final yang dipakai saat ini: `{shape_text}`")
    lines.append(f"- Model terbaik pada evaluasi holdout utama: `{best_primary['model']}`")
    lines.append(f"- Feature set utama yang dipakai: `{strategy['label']}`")
    lines.append(f"- PR-AUC: `{float(best_primary['pr_auc']):.4f}`")
    lines.append(f"- Recall: `{float(best_primary['recall']):.4f}`")
    lines.append(f"- Precision: `{float(best_primary['precision']):.4f}`")
    lines.append(f"- F1-score: `{float(best_primary['f1']):.4f}`")
    lines.append("")
    lines.append("## Data yang Dipakai")
    lines.append("")
    lines.append(f"- Model final memakai `{strategy['label']}`.")
    lines.append(f"- Deskripsi singkat: {strategy['description']}")
    lines.append(f"- Alasan memilih set ini: {rationale_text}")
    lines.append("- State, physician specialty, diagnosis ICD, medication, PracticeGuid, dan PatientGuid tidak dipakai sebagai predictor model utama.")
    lines.append("- Keputusan ini dibuat agar model tidak bergantung pada lokasi, jalur pelayanan, diagnosis yang sudah jadi, obat, atau identifier pasien.")
    lines.append("")
    lines.append("## Model yang Paling Bagus")
    lines.append("")
    lines.append(
        f"- Secara holdout, model utama terbaik adalah `{best_primary['model']}` pada `{strategy['label']}` "
        f"dengan PR-AUC `{float(best_primary['pr_auc']):.4f}` dan recall `{float(best_primary['recall']):.4f}`."
    )
    lines.append(f"- {cv_text}")
    lines.append(f"- {threshold_text}")
    lines.append("")
    lines.append("## Penjelasan XAI")
    lines.append("")
    lines.append(f"- {xai_text}")
    lines.append("- Secara umum, model final diarahkan untuk membaca usia, BMI, dan tekanan darah sebagai sinyal utama klasifikasi status T2DM.")
    lines.append("- SHAP digunakan untuk menjelaskan kontribusi model, bukan untuk membuat klaim sebab-akibat.")
    lines.append("")
    lines.append("## Ringkasan Keputusan")
    lines.append("")
    lines.append(f"- Model utama untuk penulisan: `{best_primary['model']}`")
    lines.append(f"- Feature set utama untuk penulisan: `{strategy['label']}`")
    lines.append(f"- {threshold_text}")
    lines.append("- Strategi imbalance: `weighted`")
    lines.append("- SMOTE: tidak dipakai sebagai default")
    lines.append("- XAI: menggunakan SHAP pada model final")
    return "\n".join(lines)


def get_dataset_shape_text(prepare_report_path: Path) -> str:
    if not prepare_report_path.exists():
        return "belum tersedia"
    content = prepare_report_path.read_text(encoding="utf-8")
    match = re.search(r"Final shape: `([^`]+)`", content)
    return match.group(1) if match else "belum tersedia"


def get_cv_summary_text(cv_path: Path) -> str:
    if not cv_path.exists():
        return "Hasil cross-validation belum tersedia."

    cv_df = pd.read_csv(cv_path)
    primary_df = cv_df[cv_df["feature_set_role"] == "primary"].copy() if "feature_set_role" in cv_df.columns else cv_df
    if primary_df.empty:
        return "Hasil cross-validation belum tersedia."

    best_cv = primary_df.iloc[0]
    return (
        f"Secara cross-validation, model paling stabil saat ini adalah `{best_cv['model']}` pada "
        f"`{best_cv['feature_set_label']}` dengan mean PR-AUC `{float(best_cv['mean_pr_auc']):.4f}` "
        f"dan mean F1 `{float(best_cv['mean_f1']):.4f}`."
    )


def get_threshold_summary_text(threshold_path: Path, feature_set_name: str, model_name: str) -> str:
    if not threshold_path.exists():
        return "Hasil threshold tuning belum tersedia."

    threshold_df = pd.read_csv(threshold_path)
    target_df = threshold_df[
        (threshold_df["feature_set"] == feature_set_name)
        & (threshold_df["model"] == model_name)
    ].copy()
    if target_df.empty:
        return "Hasil threshold tuning untuk model utama belum tersedia."

    row = target_df.iloc[0]
    return (
        f"Threshold terbaik untuk model utama saat ini adalah `{float(row['threshold']):.2f}` "
        f"dengan recall `{float(row['recall']):.4f}` dan precision `{float(row['precision']):.4f}`."
    )


def get_xai_summary_text(shap_path: Path, top_n: int = 5) -> str:
    if not shap_path.exists():
        return "Hasil XAI belum tersedia."

    shap_df = pd.read_csv(shap_path).head(top_n)
    if shap_df.empty:
        return "Hasil XAI belum tersedia."

    top_features = ", ".join(str(feature) for feature in shap_df["feature"].tolist())
    return f"Fitur SHAP paling dominan pada model final adalah: {top_features}."


def get_feature_set_rationale(feature_set_name: str) -> str:
    rationales = {
        "clinical_core": "set ini paling sederhana dan hanya memakai fitur klinis yang mudah dijelaskan.",
        "clinical_core_extreme": "set ini menambah nilai maksimum BMI dan tekanan darah untuk menangkap kondisi ekstrem tanpa menambah fitur proxy pelayanan.",
        "clinical_core_extreme_weight": "set ini menguji apakah berat badan mentah masih menambah informasi setelah BMI digunakan.",
        "full_transcript_comparator": "set ini dipakai sebagai pembanding kompleks karena memakai transcript lebih lengkap, tetapi tetap tanpa State, physician, ICD, medication, atau ID.",
    }
    return rationales.get(feature_set_name, "set ini dipilih berdasarkan performa dan kesesuaian metodologis.")


def open_artifact(path: Path) -> None:
    if not path.exists():
        print_message(f"File belum tersedia: {path}")
        pause()
        return

    try:
        if sys.platform.startswith("win") and hasattr(os, "startfile"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            webbrowser.open(path.resolve().as_uri())
        print_message(f"Membuka file: {path}")
    except Exception as exc:  # noqa: BLE001
        print_message(f"Gagal membuka file otomatis: {exc}")
        print_message(f"Silakan buka manual dari path berikut:\n{path}")
    pause()


def prompt_menu(title: str, options: list[tuple[str, str]]) -> str:
    render_header(title)
    for key, label in options:
        print_message(f"[{key}] {label}", is_line=True)
    return input("\nPilih menu: ").strip()


def render_header(title: str) -> None:
    if HAS_RICH and Panel is not None:
        console.print(
            Panel(
                f"[bold]{APP_TITLE}[/bold]\n{APP_SUBTITLE}\n\n[cyan]{title}[/cyan]",
                expand=True,
            )
        )
    else:
        print("=" * 64)
        print(APP_TITLE)
        print(APP_SUBTITLE)
        print("-" * 64)
        print(title)
        print("=" * 64)


def print_message(message: str, is_line: bool = False) -> None:
    if HAS_RICH and console is not None:
        console.print(message)
    else:
        if not is_line and not message.endswith("\n"):
            print(message)
        else:
            print(message)


def print_block(title: str, content: str) -> None:
    if HAS_RICH and Panel is not None:
        console.print(Panel(content, title=title, expand=True))
    else:
        print("=" * 72)
        print(title)
        print("=" * 72)
        print(content)


def pause() -> None:
    input("\nTekan Enter untuk melanjutkan...")
