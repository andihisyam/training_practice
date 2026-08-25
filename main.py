from __future__ import annotations

import argparse
import sys
from pathlib import Path

from practicefusion.config import APP_FINAL_DATASET_PATH, EDA_OUTPUT_DIR, PREPARE_OUTPUT_DIR, TRAIN_OUTPUT_DIR
from practicefusion.interactive_menu import run_interactive_menu
from practicefusion.pipelines.eda import run_eda
from practicefusion.pipelines.prepare_data import run_prepare_data
from practicefusion.pipelines.train import (
    run_error_analysis,
    run_feature_leakage_audit,
    run_leakage_check,
    run_main_cv,
    run_smote_comparison,
    run_threshold_tuning,
    run_training,
    run_weighting_comparison,
    run_xai_analysis,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Main entrypoint untuk alur riset PracticeFusion: prepare data, EDA, dan training baseline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Bangun final dataset dari raw source.")
    prepare.add_argument("--dataset-out", type=Path, default=APP_FINAL_DATASET_PATH, help="Lokasi output final dataset CSV.")
    prepare.add_argument("--transcript-out", type=Path, default=PREPARE_OUTPUT_DIR / "transcript_rebuilt.csv", help="Lokasi output transcript rebuilt CSV.")

    eda = subparsers.add_parser("eda", help="Jalankan EDA terarah pada final dataset.")
    eda.add_argument("--dataset", type=Path, default=APP_FINAL_DATASET_PATH, help="Path final dataset CSV.")
    eda.add_argument("--out-dir", type=Path, default=EDA_OUTPUT_DIR, help="Folder output report dan plot EDA.")

    train = subparsers.add_parser("train", help="Jalankan baseline training pada final dataset.")
    train.add_argument("--dataset", type=Path, default=APP_FINAL_DATASET_PATH, help="Path final dataset CSV.")
    train.add_argument("--out-dir", type=Path, default=TRAIN_OUTPUT_DIR, help="Folder output hasil training.")
    train.add_argument(
        "--feature-sets",
        nargs="*",
        default=None,
        choices=["demografi_transcript", "demografi_transcript_physician", "diagnosis", "diagnosis_physician", "full"],
        help="Feature set yang ingin dijalankan. `full` diposisikan sebagai pembanding karena masih mengandung medication.",
    )
    train.add_argument(
        "--models",
        nargs="*",
        default=None,
        choices=["Logistic Regression", "SVM", "KNN", "Gradient Boosting", "XGBoost", "LightGBM"],
        help="Model yang ingin dijalankan. Kosong berarti semua.",
    )
    train.add_argument("--test-size", type=float, default=0.3, help="Proporsi test split.")
    train.add_argument("--random-state", type=int, default=42, help="Random state untuk split.")

    leakage = subparsers.add_parser("leakage-check", help="Jalankan eksperimen leakage check pada final dataset.")
    leakage.add_argument("--dataset", type=Path, default=APP_FINAL_DATASET_PATH, help="Path final dataset CSV.")
    leakage.add_argument("--out-dir", type=Path, default=TRAIN_OUTPUT_DIR, help="Folder output hasil leakage check.")
    leakage.add_argument(
        "--models",
        nargs="*",
        default=None,
        choices=["Logistic Regression", "SVM", "KNN", "Gradient Boosting", "XGBoost", "LightGBM"],
        help="Model yang ingin dipakai untuk leakage check. Kosong berarti model default leakage check.",
    )
    leakage.add_argument("--test-size", type=float, default=0.3, help="Proporsi test split.")
    leakage.add_argument("--random-state", type=int, default=42, help="Random state untuk split.")

    leakage_audit = subparsers.add_parser("leakage-audit", help="Audit fitur diagnosis, physician, dan medication yang berpotensi leakage.")
    leakage_audit.add_argument("--dataset", type=Path, default=APP_FINAL_DATASET_PATH, help="Path final dataset CSV.")
    leakage_audit.add_argument("--out-dir", type=Path, default=TRAIN_OUTPUT_DIR, help="Folder output audit leakage.")
    leakage_audit.add_argument("--min-support", type=int, default=10, help="Support minimum untuk daftar medication berasosiasi tinggi.")

    smote = subparsers.add_parser("smote-compare", help="Bandingkan training tanpa SMOTE vs dengan SMOTE pada feature set yang dipilih.")
    smote.add_argument("--dataset", type=Path, default=APP_FINAL_DATASET_PATH, help="Path final dataset CSV.")
    smote.add_argument("--out-dir", type=Path, default=TRAIN_OUTPUT_DIR, help="Folder output hasil SMOTE comparison.")
    smote.add_argument(
        "--feature-sets",
        nargs="*",
        default=None,
        choices=["demografi_transcript", "demografi_transcript_physician", "diagnosis", "diagnosis_physician", "full"],
        help="Feature set yang ingin diuji pada eksperimen SMOTE. Kosong berarti semua 4 set.",
    )
    smote.add_argument(
        "--models",
        nargs="*",
        default=None,
        choices=["Logistic Regression", "SVM", "KNN", "Gradient Boosting", "XGBoost", "LightGBM"],
        help="Model yang ingin diuji pada eksperimen SMOTE. Kosong berarti model default SMOTE comparison.",
    )
    smote.add_argument("--test-size", type=float, default=0.3, help="Proporsi test split.")
    smote.add_argument("--random-state", type=int, default=42, help="Random state untuk split.")

    weighting = subparsers.add_parser("weight-compare", help="Bandingkan model weighted vs unweighted pada feature set yang dipilih.")
    weighting.add_argument("--dataset", type=Path, default=APP_FINAL_DATASET_PATH, help="Path final dataset CSV.")
    weighting.add_argument("--out-dir", type=Path, default=TRAIN_OUTPUT_DIR, help="Folder output hasil weighting comparison.")
    weighting.add_argument(
        "--feature-sets",
        nargs="*",
        default=None,
        choices=["demografi_transcript", "demografi_transcript_physician", "diagnosis", "diagnosis_physician", "full"],
        help="Feature set yang ingin diuji pada eksperimen weighting. Kosong berarti semua 4 set.",
    )
    weighting.add_argument(
        "--models",
        nargs="*",
        default=None,
        choices=["Logistic Regression", "SVM", "KNN", "Gradient Boosting", "XGBoost", "LightGBM"],
        help="Model yang ingin diuji pada eksperimen weighting. Kosong berarti model default weighting comparison.",
    )
    weighting.add_argument("--test-size", type=float, default=0.3, help="Proporsi test split.")
    weighting.add_argument("--random-state", type=int, default=42, help="Random state untuk split.")

    threshold = subparsers.add_parser("threshold-tune", help="Cari threshold terbaik untuk model kandidat pada feature set yang dipilih.")
    threshold.add_argument("--dataset", type=Path, default=APP_FINAL_DATASET_PATH, help="Path final dataset CSV.")
    threshold.add_argument("--out-dir", type=Path, default=TRAIN_OUTPUT_DIR, help="Folder output hasil threshold tuning.")
    threshold.add_argument(
        "--feature-sets",
        nargs="*",
        default=None,
        choices=["demografi_transcript", "demografi_transcript_physician", "diagnosis", "diagnosis_physician", "full"],
        help="Feature set yang ingin diuji. Jika kosong dan model juga kosong, dipakai 2 eksperimen default kandidat utama.",
    )
    threshold.add_argument(
        "--models",
        nargs="*",
        default=None,
        choices=["Logistic Regression", "SVM", "KNN", "Gradient Boosting", "XGBoost", "LightGBM"],
        help="Model yang ingin diuji. Jika kosong dan feature set juga kosong, dipakai 2 eksperimen default kandidat utama.",
    )
    threshold.add_argument("--thresholds", nargs="*", type=float, default=None, help="Daftar threshold, misalnya 0.2 0.3 0.4 0.5.")
    threshold.add_argument("--test-size", type=float, default=0.3, help="Proporsi test split.")
    threshold.add_argument("--random-state", type=int, default=42, help="Random state untuk split.")

    cv_main = subparsers.add_parser("cv-main", help="Jalankan cross-validation utama pada feature set primary.")
    cv_main.add_argument("--dataset", type=Path, default=APP_FINAL_DATASET_PATH, help="Path final dataset CSV.")
    cv_main.add_argument("--out-dir", type=Path, default=TRAIN_OUTPUT_DIR, help="Folder output hasil cross-validation utama.")
    cv_main.add_argument(
        "--feature-sets",
        nargs="*",
        default=None,
        choices=["demografi_transcript", "demografi_transcript_physician", "diagnosis", "diagnosis_physician", "full"],
        help="Feature set yang ingin diuji pada CV. Kosong berarti 3 feature set primary.",
    )
    cv_main.add_argument(
        "--models",
        nargs="*",
        default=None,
        choices=["Logistic Regression", "SVM", "KNN", "Gradient Boosting", "XGBoost", "LightGBM"],
        help="Model yang ingin diuji pada CV. Kosong berarti semua model.",
    )
    cv_main.add_argument("--n-splits", type=int, default=5, help="Jumlah fold untuk Stratified K-Fold.")
    cv_main.add_argument("--random-state", type=int, default=42, help="Random state untuk fold shuffle.")

    error_analysis = subparsers.add_parser("error-analysis", help="Analisis false positive dan false negative pada model kandidat.")
    error_analysis.add_argument("--dataset", type=Path, default=APP_FINAL_DATASET_PATH, help="Path final dataset CSV.")
    error_analysis.add_argument("--out-dir", type=Path, default=TRAIN_OUTPUT_DIR, help="Folder output hasil error analysis.")
    error_analysis.add_argument(
        "--feature-set",
        type=str,
        default="demografi_transcript_physician",
        choices=["demografi_transcript", "demografi_transcript_physician", "diagnosis", "diagnosis_physician", "full"],
        help="Feature set yang ingin dianalisis.",
    )
    error_analysis.add_argument(
        "--model",
        type=str,
        default="XGBoost",
        choices=["Logistic Regression", "SVM", "KNN", "Gradient Boosting", "XGBoost", "LightGBM"],
        help="Model yang ingin dianalisis.",
    )
    error_analysis.add_argument("--thresholds", nargs="*", type=float, default=None, help="Daftar threshold untuk dibandingkan, misalnya 0.5 0.6.")
    error_analysis.add_argument("--test-size", type=float, default=0.3, help="Proporsi test split.")
    error_analysis.add_argument("--random-state", type=int, default=42, help="Random state untuk split.")

    xai = subparsers.add_parser("xai", help="Jalankan interpretasi model terbaik menggunakan SHAP.")
    xai.add_argument("--dataset", type=Path, default=APP_FINAL_DATASET_PATH, help="Path final dataset CSV.")
    xai.add_argument("--out-dir", type=Path, default=TRAIN_OUTPUT_DIR, help="Folder output hasil XAI.")
    xai.add_argument(
        "--feature-set",
        type=str,
        default="demografi_transcript_physician",
        choices=["demografi_transcript", "demografi_transcript_physician", "diagnosis", "diagnosis_physician", "full"],
        help="Feature set yang ingin diinterpretasikan.",
    )
    xai.add_argument(
        "--model",
        type=str,
        default="XGBoost",
        choices=["Logistic Regression", "SVM", "KNN", "Gradient Boosting", "XGBoost", "LightGBM"],
        help="Model yang ingin diinterpretasikan.",
    )
    xai.add_argument("--threshold", type=float, default=0.5, help="Threshold klasifikasi yang dipakai untuk local explanation.")
    xai.add_argument("--test-size", type=float, default=0.3, help="Proporsi test split.")
    xai.add_argument("--random-state", type=int, default=42, help="Random state untuk split.")

    all_cmd = subparsers.add_parser("all", help="Jalankan prepare-data, lalu EDA, lalu training baseline.")
    all_cmd.add_argument("--dataset-out", type=Path, default=APP_FINAL_DATASET_PATH, help="Lokasi output final dataset CSV.")
    all_cmd.add_argument("--eda-out-dir", type=Path, default=EDA_OUTPUT_DIR, help="Folder output EDA.")
    all_cmd.add_argument("--train-out-dir", type=Path, default=TRAIN_OUTPUT_DIR, help="Folder output training.")
    all_cmd.add_argument("--test-size", type=float, default=0.3, help="Proporsi test split untuk training.")
    all_cmd.add_argument("--random-state", type=int, default=42, help="Random state untuk split training.")

    return parser


def main() -> int:
    if len(sys.argv) == 1:
        return run_interactive_menu()

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare-data":
        result = run_prepare_data(final_dataset_path=args.dataset_out, transcript_output_path=args.transcript_out)
        print(f"Final dataset saved to: {result['final_dataset_path']}")
        print(f"Transcript rebuild saved to: {result['transcript_output_path']}")
        return 0

    if args.command == "eda":
        result = run_eda(dataset_path=args.dataset, out_dir=args.out_dir)
        print(f"EDA report saved to: {result['report_path']}")
        print(f"EDA summary saved to: {result['summary_path']}")
        return 0

    if args.command == "train":
        result = run_training(
            dataset_path=args.dataset,
            out_dir=args.out_dir,
            selected_feature_sets=args.feature_sets,
            selected_models=args.models,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(f"Training results saved to: {result['results_path']}")
        print(f"Best model saved to: {result['best_model_path']}")
        return 0

    if args.command == "leakage-check":
        result = run_leakage_check(
            dataset_path=args.dataset,
            out_dir=args.out_dir,
            selected_models=args.models,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(f"Leakage check results saved to: {result['results_path']}")
        print(f"Leakage check summary saved to: {result['summary_path']}")
        return 0

    if args.command == "leakage-audit":
        result = run_feature_leakage_audit(
            dataset_path=args.dataset,
            out_dir=args.out_dir,
            min_support=args.min_support,
        )
        print(f"Leakage audit report saved to: {result['report_path']}")
        print(f"Leakage audit summary saved to: {result['summary_path']}")
        return 0

    if args.command == "smote-compare":
        result = run_smote_comparison(
            dataset_path=args.dataset,
            out_dir=args.out_dir,
            selected_feature_sets=args.feature_sets,
            selected_models=args.models,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(f"SMOTE comparison results saved to: {result['results_path']}")
        print(f"SMOTE comparison summary saved to: {result['summary_path']}")
        return 0

    if args.command == "weight-compare":
        result = run_weighting_comparison(
            dataset_path=args.dataset,
            out_dir=args.out_dir,
            selected_feature_sets=args.feature_sets,
            selected_models=args.models,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(f"Weighting comparison results saved to: {result['results_path']}")
        print(f"Weighting comparison summary saved to: {result['summary_path']}")
        return 0

    if args.command == "threshold-tune":
        result = run_threshold_tuning(
            dataset_path=args.dataset,
            out_dir=args.out_dir,
            selected_feature_sets=args.feature_sets,
            selected_models=args.models,
            thresholds=args.thresholds,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(f"Threshold tuning results saved to: {result['results_path']}")
        print(f"Threshold tuning summary saved to: {result['summary_path']}")
        return 0

    if args.command == "cv-main":
        result = run_main_cv(
            dataset_path=args.dataset,
            out_dir=args.out_dir,
            selected_feature_sets=args.feature_sets,
            selected_models=args.models,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        print(f"Main CV fold results saved to: {result['results_path']}")
        print(f"Main CV summary saved to: {result['summary_path']}")
        return 0

    if args.command == "error-analysis":
        result = run_error_analysis(
            dataset_path=args.dataset,
            out_dir=args.out_dir,
            feature_set_name=args.feature_set,
            model_name=args.model,
            thresholds=args.thresholds,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(f"Error analysis summary saved to: {result['summary_path']}")
        print(f"Error analysis report saved to: {result['report_path']}")
        return 0

    if args.command == "xai":
        result = run_xai_analysis(
            dataset_path=args.dataset,
            out_dir=args.out_dir,
            feature_set_name=args.feature_set,
            model_name=args.model,
            threshold=args.threshold,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(f"XAI report saved to: {result['report_path']}")
        print(f"SHAP summary plot saved to: {result['summary_plot_path']}")
        return 0

    if args.command == "all":
        prepare_result = run_prepare_data(final_dataset_path=args.dataset_out)
        eda_result = run_eda(dataset_path=prepare_result["final_dataset_path"], out_dir=args.eda_out_dir)
        train_result = run_training(
            dataset_path=prepare_result["final_dataset_path"],
            out_dir=args.train_out_dir,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(f"Prepare-data complete: {prepare_result['final_dataset_path']}")
        print(f"EDA report: {eda_result['report_path']}")
        print(f"Training report: {train_result['report_path']}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
