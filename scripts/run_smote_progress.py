from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from practicefusion.config import APP_FINAL_DATASET_PATH, TRAIN_OUTPUT_DIR
from practicefusion.pipelines.train import (
    DEFAULT_SMOTE_MODELS,
    build_feature_sets,
    describe_feature_set,
    evaluate_experiment_with_optional_smote,
    make_model_builders,
)
from practicefusion.utils.io import ensure_dir, load_csv, save_markdown


def main() -> int:
    dataset_path = APP_FINAL_DATASET_PATH
    out_dir = TRAIN_OUTPUT_DIR
    random_state = 42
    test_size = 0.3

    ensure_dir(out_dir)
    df = load_csv(dataset_path)
    feature_sets, _ = build_feature_sets(df)

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        stratify=df["DMIndicator"],
        random_state=random_state,
    )
    y_train_ref = df.loc[train_idx, "DMIndicator"].astype(int)
    neg_pos_ratio = float((y_train_ref == 0).sum() / (y_train_ref == 1).sum())
    model_builders = make_model_builders(neg_pos_ratio=neg_pos_ratio, random_state=random_state)
    model_builders = {name: model_builders[name] for name in DEFAULT_SMOTE_MODELS}

    results: list[dict[str, object]] = []
    total = len(feature_sets) * len(model_builders) * 2
    step = 0

    partial_path = out_dir / "smote_comparison_results_partial.csv"
    for feature_set_name, feature_cols in feature_sets.items():
        strategy = describe_feature_set(feature_set_name)
        for model_name, model_builder in model_builders.items():
            for use_smote in [False, True]:
                step += 1
                sampling_strategy = "smote" if use_smote else "no_smote"
                print(
                    f"[{step}/{total}] RUN {feature_set_name} | {model_name} | {sampling_strategy} | n_features={len(feature_cols)}",
                    flush=True,
                )
                result, _ = evaluate_experiment_with_optional_smote(
                    df=df,
                    feature_set_name=feature_set_name,
                    feature_cols=feature_cols,
                    model_name=model_name,
                    model_builder=model_builder,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    use_smote=use_smote,
                    random_state=random_state,
                )
                result["experiment_key"] = f"{model_name}__{feature_set_name}__{sampling_strategy}"
                result["feature_set_label"] = strategy["label"]
                result["feature_set_role"] = strategy["role"]
                results.append(result)
                pd.DataFrame(results).to_csv(partial_path, index=False)
                print(
                    f"[{step}/{total}] DONE {feature_set_name} | {model_name} | {sampling_strategy} | "
                    f"pr_auc={result['pr_auc']:.4f} | recall={result['recall']:.4f} | precision={result['precision']:.4f}",
                    flush=True,
                )

    results_df = pd.DataFrame(results).sort_values(
        ["feature_set", "model", "sampling_strategy"], ascending=[True, True, True]
    ).reset_index(drop=True)
    results_path = out_dir / "smote_comparison_results.csv"
    results_df.to_csv(results_path, index=False)

    comparison_rows: list[dict[str, object]] = []
    for (feature_set_name, model_name), group in results_df.groupby(["feature_set", "model"]):
        group = group.set_index("sampling_strategy")
        if {"no_smote", "smote"} - set(group.index):
            continue
        no_smote = group.loc["no_smote"]
        smote = group.loc["smote"]
        strategy = describe_feature_set(feature_set_name)
        comparison_rows.append(
            {
                "feature_set": feature_set_name,
                "feature_set_label": strategy["label"],
                "feature_set_role": strategy["role"],
                "model": model_name,
                "no_smote_pr_auc": round(float(no_smote["pr_auc"]), 4),
                "smote_pr_auc": round(float(smote["pr_auc"]), 4),
                "delta_pr_auc": round(float(smote["pr_auc"] - no_smote["pr_auc"]), 4),
                "no_smote_recall": round(float(no_smote["recall"]), 4),
                "smote_recall": round(float(smote["recall"]), 4),
                "delta_recall": round(float(smote["recall"] - no_smote["recall"]), 4),
                "no_smote_precision": round(float(no_smote["precision"]), 4),
                "smote_precision": round(float(smote["precision"]), 4),
                "delta_precision": round(float(smote["precision"] - no_smote["precision"]), 4),
                "no_smote_f1": round(float(no_smote["f1"]), 4),
                "smote_f1": round(float(smote["f1"]), 4),
                "delta_f1": round(float(smote["f1"] - no_smote["f1"]), 4),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["feature_set", "delta_pr_auc", "delta_recall"], ascending=[True, False, False]
    ).reset_index(drop=True)
    comparison_path = out_dir / "smote_comparison_summary.csv"
    comparison_df.to_csv(comparison_path, index=False)

    lines: list[str] = []
    lines.append("# SMOTE Comparison Report")
    lines.append("")
    lines.append(f"- Dataset: `{dataset_path}`")
    lines.append(f"- Test size: `{test_size}`")
    lines.append(f"- Random state: `{random_state}`")
    lines.append("")
    lines.append("## Ringkasan Perbandingan")
    lines.append("")
    for row in comparison_df.to_dict(orient="records"):
        lines.append(
            f"- `{row['feature_set']}` + `{row['model']}`: "
            f"PR-AUC `{row['no_smote_pr_auc']}` -> `{row['smote_pr_auc']}` (delta `{row['delta_pr_auc']}`), "
            f"Recall `{row['no_smote_recall']}` -> `{row['smote_recall']}` (delta `{row['delta_recall']}`), "
            f"Precision `{row['no_smote_precision']}` -> `{row['smote_precision']}` (delta `{row['delta_precision']}`), "
            f"F1 `{row['no_smote_f1']}` -> `{row['smote_f1']}` (delta `{row['delta_f1']}`)"
        )
    report_path = out_dir / "smote_comparison_report.md"
    save_markdown(report_path, "\n".join(lines))

    print(f"SAVED {results_path}", flush=True)
    print(f"SAVED {comparison_path}", flush=True)
    print(f"SAVED {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
