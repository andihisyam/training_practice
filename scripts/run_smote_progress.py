from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from practicefusion.config import APP_FINAL_DATASET_PATH, TRAIN_OUTPUT_DIR
from practicefusion.pipelines.train import run_smote_comparison


def main() -> int:
    result = run_smote_comparison(dataset_path=APP_FINAL_DATASET_PATH, out_dir=TRAIN_OUTPUT_DIR)
    print(f"SMOTE comparison results saved to: {result['results_path']}")
    print(f"SMOTE comparison summary saved to: {result['summary_path']}")
    print(f"SMOTE comparison report saved to: {result['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
