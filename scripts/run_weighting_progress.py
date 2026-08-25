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
    DEFAULT_WEIGHTING_MODELS,
    build_feature_sets,
    describe_feature_set,
    evaluate_experiment,
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
    weighted_builders = make_model_builders(neg_pos_ratio=neg_pos_ratio, random_state=random_state, use_class_balancing=True)
    unweighted_builders = make_model_builders(neg_pos_ratio=neg_pos_ratio, random_state=random_state, use_class_balancing=False)

    results: list[dict[str, object]] = []
    total = len(feature_sets) * len(DEFAULT_WEIGHTING_MODELS) * 2
    step = 0

    partial_path = out_dir / "weighting_comparison_results_partial.csv"
    for feature_set_name, feature_cols in feature_sets.items():
        strategy = describe_feature_set(feature_set_name)
        for model_name in DEFAULT_WEIGHTING_MODELS:
            for weighting_mode, model_builder, use_class_balancing in [
                ("weighted", weighted_builders[model_name], True),
                ("unweighted", unweighted_builders[model_name], False),
            ]:
                step += 1
                print(
                    f"[{step}/{total}] RUN {feature_set_name} | {model_name} | {weighting_mode} | n_features={len(feature_cols)}",
                    flush=True,
                )
                result, _ = evaluate_experiment(
                    df=df,
                    feature_set_name=feature_set_name,
                    feature_cols=feature_cols,
                    model_name=model_name,
                    model_builder=model_builder,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    use_class_balancing=use_class_balancing,
                )
                result["experiment_key"] = f"{model_name}__{feature_set_name}__{weighting_mode}"
                result["feature_set_label"] = strategy["label"]
                result["feature_set_role"] = strategy["role"]
                result["weighting_mode"] = weighting_mode
                results.append(result)
                pd.DataFrame(results).to_csv(partial_path, index=False)
                print(
                    f"[{step}/{total}] DONE {feature_set_name} | {model_name} | {weighting_mode} | "
                    f"pr_auc={result['pr_auc']:.4f} | recall={result['recall']:.4f} | precision={result['precision']:.4f}",
                    flush=True,
                )

    results_df = pd.DataFrame(results).sort_values(
        ["feature_set", "model", "weighting_mode"], ascending=[True, True, True]
    ).reset_index(drop=True)
    results_path = out_dir / "weighting_comparison_results.csv"
    results_df.to_csv(results_path, index=False)

    comparison_rows: list[dict[str, object]] = []
    for (feature_set_name, model_name), group in results_df.groupby(["feature_set", "model"]):
        group = group.set_index("weighting_mode")
        if {"weighted", "unweighted"} - set(group.index):
            continue
        weighted = group.loc["weighted"]
        unweighted = group.loc["unweighted"]
        strategy = describe_feature_set(feature_set_name)
        comparison_rows.append(
            {
                "feature_set": feature_set_name,
                "feature_set_label": strategy["label"],
                "feature_set_role": strategy["role"],
                "model": model_name,
                "weighted_pr_auc": round(float(weighted["pr_auc"]), 4),
                "unweighted_pr_auc": round(float(unweighted["pr_auc"]), 4),
                "delta_pr_auc_weighted_minus_unweighted": round(float(weighted["pr_auc"] - unweighted["pr_auc"]), 4),
                "weighted_recall": round(float(weighted["recall"]), 4),
                "unweighted_recall": round(float(unweighted["recall"]), 4),
                "delta_recall_weighted_minus_unweighted": round(float(weighted["recall"] - unweighted["recall"]), 4),
                "weighted_precision": round(float(weighted["precision"]), 4),
                "unweighted_precision": round(float(unweighted["precision"]), 4),
                "delta_precision_weighted_minus_unweighted": round(float(weighted["precision"] - unweighted["precision"]), 4),
                "weighted_f1": round(float(weighted["f1"]), 4),
                "unweighted_f1": round(float(unweighted["f1"]), 4),
                "delta_f1_weighted_minus_unweighted": round(float(weighted["f1"] - unweighted["f1"]), 4),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["feature_set", "delta_pr_auc_weighted_minus_unweighted", "delta_recall_weighted_minus_unweighted"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    comparison_path = out_dir / "weighting_comparison_summary.csv"
    comparison_df.to_csv(comparison_path, index=False)

    lines: list[str] = []
    lines.append("# Weighting Comparison Report")
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
            f"PR-AUC weighted `{row['weighted_pr_auc']}` vs unweighted `{row['unweighted_pr_auc']}` "
            f"(delta `{row['delta_pr_auc_weighted_minus_unweighted']}`), "
            f"Recall weighted `{row['weighted_recall']}` vs unweighted `{row['unweighted_recall']}` "
            f"(delta `{row['delta_recall_weighted_minus_unweighted']}`), "
            f"Precision weighted `{row['weighted_precision']}` vs unweighted `{row['unweighted_precision']}` "
            f"(delta `{row['delta_precision_weighted_minus_unweighted']}`), "
            f"F1 weighted `{row['weighted_f1']}` vs unweighted `{row['unweighted_f1']}` "
            f"(delta `{row['delta_f1_weighted_minus_unweighted']}`)"
        )
    report_path = out_dir / "weighting_comparison_report.md"
    save_markdown(report_path, "\n".join(lines))

    print(f"SAVED {results_path}", flush=True)
    print(f"SAVED {comparison_path}", flush=True)
    print(f"SAVED {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
