from __future__ import annotations

from pathlib import Path

from practicefusion.config import APP_FINAL_DATASET_PATH, EDA_OUTPUT_DIR, TRAIN_OUTPUT_DIR
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


def main() -> int:
    print("[1/10] Running prepare-data...")
    prepare_result = run_prepare_data()
    dataset_path: Path = prepare_result["final_dataset_path"]

    print("[2/10] Running EDA...")
    run_eda(dataset_path=dataset_path, out_dir=EDA_OUTPUT_DIR)

    print("[3/10] Running baseline training...")
    run_training(dataset_path=dataset_path, out_dir=TRAIN_OUTPUT_DIR)

    print("[4/10] Running leakage audit...")
    run_feature_leakage_audit(dataset_path=dataset_path, out_dir=TRAIN_OUTPUT_DIR)

    print("[5/10] Running leakage check...")
    run_leakage_check(dataset_path=dataset_path, out_dir=TRAIN_OUTPUT_DIR)

    print("[6/10] Running weighting comparison...")
    run_weighting_comparison(dataset_path=dataset_path, out_dir=TRAIN_OUTPUT_DIR)

    print("[7/10] Running SMOTE comparison...")
    run_smote_comparison(dataset_path=dataset_path, out_dir=TRAIN_OUTPUT_DIR)

    print("[8/10] Running main CV...")
    run_main_cv(dataset_path=dataset_path, out_dir=TRAIN_OUTPUT_DIR)

    print("[9/10] Running threshold tuning and error analysis...")
    run_threshold_tuning(dataset_path=dataset_path, out_dir=TRAIN_OUTPUT_DIR)
    run_error_analysis(dataset_path=dataset_path, out_dir=TRAIN_OUTPUT_DIR, thresholds=[0.50, 0.60])

    print("[10/10] Running final XAI...")
    run_xai_analysis(dataset_path=dataset_path, out_dir=TRAIN_OUTPUT_DIR, threshold=0.50)

    print("Clean end-to-end pipeline finished.")
    print(f"Final dataset: {APP_FINAL_DATASET_PATH}")
    print(f"EDA outputs: {EDA_OUTPUT_DIR}")
    print(f"Training/XAI outputs: {TRAIN_OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
