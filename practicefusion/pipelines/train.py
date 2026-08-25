from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from practicefusion.config import APP_FINAL_DATASET_PATH, TRAIN_OUTPUT_DIR
from practicefusion.utils.io import ensure_dir, load_csv, save_json, save_markdown


warnings.filterwarnings("ignore", category=ConvergenceWarning)

SCALE_SENSITIVE_MODELS = {"Logistic Regression", "SVM", "KNN"}
DEFAULT_SMOTE_MODELS = ["Logistic Regression", "Gradient Boosting", "XGBoost", "LightGBM"]
DEFAULT_WEIGHTING_MODELS = ["Logistic Regression", "SVM", "Gradient Boosting", "XGBoost", "LightGBM"]
DEFAULT_CV_FEATURE_SETS = ["demografi_transcript", "demografi_transcript_physician"]
DEFAULT_ERROR_ANALYSIS_FEATURE_SET = "demografi_transcript_physician"
DEFAULT_ERROR_ANALYSIS_MODEL = "XGBoost"
DEFAULT_ERROR_ANALYSIS_THRESHOLDS = [0.50, 0.60]
DEFAULT_XAI_FEATURE_SET = "demografi_transcript_physician"
DEFAULT_XAI_MODEL = "XGBoost"
DEFAULT_XAI_THRESHOLD = 0.50
DEFAULT_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
DEFAULT_THRESHOLD_EXPERIMENTS = [
    ("demografi_transcript_physician", "XGBoost"),
    ("demografi_transcript", "LightGBM"),
]
FEATURE_SET_STRATEGY = {
    "demografi_transcript": {
        "label": "Set A - Demografi + Transcript",
        "role": "primary",
        "description": "Baseline paling bersih, berisi demografi dan kondisi klinis dari transcript.",
    },
    "demografi_transcript_physician": {
        "label": "Set B - Demografi + Transcript + Physician",
        "role": "primary",
        "description": "Model utama tanpa ICD, dengan tambahan specialty dokter sebagai konteks layanan.",
    },
    "diagnosis": {
        "label": "Set C - Demografi + Transcript + Diagnosis",
        "role": "sensitivity",
        "description": "Set sensitivitas yang menambahkan blok ICD untuk mengukur seberapa besar model bergantung pada diagnosis.",
    },
    "diagnosis_physician": {
        "label": "Set D - Demografi + Transcript + Diagnosis + Physician",
        "role": "sensitivity",
        "description": "Set sensitivitas dengan ICD dan physician untuk membaca kenaikan performa dari fitur yang lebih dekat ke proses diagnosis.",
    },
    "full": {
        "label": "Set E - Full + Medication",
        "role": "comparator",
        "description": "Model pembanding yang tetap memasukkan medication untuk mengukur potensi leakage.",
    },
}
DIRECT_DIABETES_MEDICATION_TERMS = [
    "metformin",
    "insulin",
    "glipizide",
    "glyburide",
    "glimepiride",
    "pioglitazone",
    "rosiglitazone",
    "sitagliptin",
    "liraglutide",
    "exenatide",
    "canagliflozin",
    "dapagliflozin",
    "empagliflozin",
    "repaglinide",
    "nateglinide",
    "acarbose",
    "miglitol",
    "tolazamide",
    "tolbutamide",
    "chlorpropamide",
]
DIABETES_SUPPLY_TERMS = [
    "glucose",
    "strip",
    "glucometer",
    "lancet",
    "swab",
    "alcohol",
    "pad",
    "pads",
    "isopropyl",
    "needle",
    "syringe",
    "pump",
    "onetouch",
    "freestyle",
    "accuchek",
    "accucheck",
]


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def is_binary_like_series(series: pd.Series) -> bool:
    values = set(pd.Series(series.dropna().unique()).tolist())
    return len(values) > 0 and values.issubset({0, 1, 0.0, 1.0})


def build_feature_sets(df: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    demographic_categorical_cols = [c for c in ["Gender", "State"] if c in df.columns]
    demographic_numeric_cols = [c for c in ["Age"] if c in df.columns]
    transcript_prefixes = ["Height", "Weight", "BMI", "SystolicBP", "DiastolicBP", "RespiratoryRate", "Temperature"]
    transcript_cols = [c for c in df.columns if any(c.startswith(f"{prefix}_") for prefix in transcript_prefixes)]
    diagnosis_cols = [
        c for c in df.columns if c.startswith("Icd9_") or c in ["DiagnosisCount", "VisitCount", "DiagnosisFreq", "AcuteCount", "AcuteFreq"]
    ]
    physician_cols = [c for c in df.columns if c.startswith("PhySp_")]
    excluded = set(
        ["PatientGuid", "DMIndicator", "Gender", "State", "Age", "YearOfBirth", "PracticeGuid"]
        + transcript_cols
        + diagnosis_cols
        + physician_cols
    )
    medication_cols = [c for c in df.columns if c not in excluded]

    all_candidate_cols = list(
        dict.fromkeys(
            demographic_categorical_cols
            + demographic_numeric_cols
            + transcript_cols
            + diagnosis_cols
            + physician_cols
            + medication_cols
        )
    )
    constant_cols = [c for c in all_candidate_cols if df[c].nunique(dropna=False) <= 1]

    feature_sets = {
        "demografi_transcript": demographic_categorical_cols + demographic_numeric_cols + transcript_cols,
        "demografi_transcript_physician": demographic_categorical_cols + demographic_numeric_cols + transcript_cols + physician_cols,
        "diagnosis": demographic_categorical_cols + demographic_numeric_cols + transcript_cols + diagnosis_cols,
        "diagnosis_physician": demographic_categorical_cols + demographic_numeric_cols + transcript_cols + diagnosis_cols + physician_cols,
        "full": demographic_categorical_cols + demographic_numeric_cols + transcript_cols + diagnosis_cols + physician_cols + medication_cols,
    }
    feature_sets = {name: [c for c in list(dict.fromkeys(cols)) if c not in constant_cols] for name, cols in feature_sets.items()}

    metadata = {
        "categorical": demographic_categorical_cols,
        "numeric_demographic": demographic_numeric_cols,
        "transcript": transcript_cols,
        "diagnosis": diagnosis_cols,
        "physician": physician_cols,
        "medication": medication_cols,
        "constant_removed": constant_cols,
    }
    return feature_sets, metadata


def build_leakage_feature_sets(metadata: dict[str, list[str]]) -> dict[str, list[str]]:
    clean_core = metadata["categorical"] + metadata["numeric_demographic"] + metadata["transcript"]
    diagnosis = metadata["diagnosis"]
    physician = metadata["physician"]
    medication = metadata["medication"]
    constant_removed = set(metadata["constant_removed"])

    leakage_sets = {
        "clean_core": clean_core,
        "clean_plus_physician": clean_core + physician,
        "with_diagnosis": clean_core + diagnosis,
        "with_medication": clean_core + medication,
        "with_diagnosis_physician": clean_core + diagnosis + physician,
        "with_physician_medication": clean_core + physician + medication,
        "full": clean_core + diagnosis + physician + medication,
    }
    return {
        name: [col for col in list(dict.fromkeys(cols)) if col not in constant_removed]
        for name, cols in leakage_sets.items()
    }


def split_feature_roles(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str], list[str]]:
    numeric_cols = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    binary_like_cols = [c for c in numeric_cols if is_binary_like_series(df[c])]
    nonbinary_numeric_cols = [c for c in numeric_cols if c not in binary_like_cols]
    return categorical_cols, binary_like_cols, nonbinary_numeric_cols


def build_preprocessor(df: pd.DataFrame, feature_cols: list[str], model_name: str) -> tuple[ColumnTransformer, dict[str, list[str]]]:
    categorical_cols, binary_like_cols, nonbinary_numeric_cols = split_feature_roles(df, feature_cols)
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if nonbinary_numeric_cols:
        steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if model_name in SCALE_SENSITIVE_MODELS:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("num_nonbinary", Pipeline(steps), nonbinary_numeric_cols))

    if binary_like_cols:
        transformers.append(("num_binary", Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]), binary_like_cols))

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", make_ohe())]),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
    role_info = {
        "categorical_cols": categorical_cols,
        "binary_like_cols": binary_like_cols,
        "nonbinary_numeric_cols": nonbinary_numeric_cols,
    }
    return preprocessor, role_info


def get_positive_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    if hasattr(pipeline, "decision_function"):
        raw_scores = pipeline.decision_function(X)
        return 1 / (1 + np.exp(-raw_scores))
    raise AttributeError("Model does not provide predict_proba or decision_function.")


def get_model_fit_kwargs(model_name: str, y_train: pd.Series, use_class_balancing: bool = True) -> dict[str, Any]:
    if model_name == "Gradient Boosting" and use_class_balancing:
        return {"model__sample_weight": compute_sample_weight(class_weight="balanced", y=y_train)}
    return {}


def import_smote() -> Any:
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError(
            "SMOTE comparison membutuhkan package `imbalanced-learn`. "
            "Install dulu dengan `pip install imbalanced-learn`, lalu jalankan ulang command SMOTE."
        ) from exc
    return SMOTE


def import_shap() -> Any:
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "Tahap XAI membutuhkan package `shap`. "
            "Install dulu dengan `pip install shap`, lalu jalankan ulang command XAI."
        ) from exc
    return shap


def describe_feature_set(feature_set_name: str) -> dict[str, str]:
    return FEATURE_SET_STRATEGY.get(
        feature_set_name,
        {
            "label": feature_set_name,
            "role": "unknown",
            "description": "Tidak ada deskripsi tambahan.",
        },
    )


def clean_feature_name(feature_name: str) -> str:
    if "__" in feature_name:
        return feature_name.split("__", 1)[1]
    return feature_name


def get_preprocessed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    if hasattr(preprocessor, "get_feature_names_out"):
        return [clean_feature_name(str(name)) for name in preprocessor.get_feature_names_out()]

    feature_names: list[str] = []
    for _, transformer, columns in preprocessor.transformers_:
        if transformer == "drop":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            try:
                names = transformer.get_feature_names_out(columns)
            except TypeError:
                names = transformer.get_feature_names_out()
            feature_names.extend(clean_feature_name(str(name)) for name in names)
        else:
            feature_names.extend(str(column) for column in columns)
    return feature_names


def normalize_shap_output(shap_output: Any) -> tuple[np.ndarray, float]:
    values = shap_output
    base_values = None

    if hasattr(shap_output, "values"):
        values = shap_output.values
        base_values = getattr(shap_output, "base_values", None)

    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]

    if base_values is None:
        base_value = 0.0
    else:
        base_array = np.asarray(base_values)
        if base_array.ndim == 0:
            base_value = float(base_array)
        elif base_array.ndim == 1:
            base_value = float(base_array[-1]) if base_array.size > 1 else float(base_array[0])
        else:
            base_value = float(np.asarray(base_array).reshape(-1)[-1])
    return values, base_value


def compute_tree_shap_values(
    shap: Any,
    model_name: str,
    model: Any,
    X_test_prepared: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, float]:
    if model_name == "XGBoost":
        dmatrix = xgb.DMatrix(X_test_prepared, feature_names=feature_names)
        contribution_matrix = model.get_booster().predict(dmatrix, pred_contribs=True)
        contribution_matrix = np.asarray(contribution_matrix)
        return contribution_matrix[:, :-1], float(contribution_matrix[0, -1])

    explainer = shap.TreeExplainer(model)
    raw_shap_output = explainer.shap_values(X_test_prepared)
    return normalize_shap_output(raw_shap_output)


def select_representative_cases(predictions_df: pd.DataFrame, threshold: float) -> list[dict[str, Any]]:
    case_rules = [
        ("TP", "tp_high_confidence", False),
        ("FN", "fn_borderline", False),
        ("FP", "fp_high_confidence", False),
        ("TN", "tn_borderline", False),
    ]
    selected_cases: list[dict[str, Any]] = []
    used_indices: set[Any] = set()

    for error_type, case_label, ascending in case_rules:
        subset = predictions_df[predictions_df["error_type"] == error_type].copy()
        if subset.empty:
            continue

        if error_type == "FN":
            subset = subset.sort_values("y_score", ascending=False)
        elif error_type == "TN":
            subset = subset.sort_values("y_score", ascending=False)
        else:
            subset = subset.sort_values("y_score", ascending=ascending)

        for _, row in subset.iterrows():
            sample_id = row["sample_index"]
            if sample_id in used_indices:
                continue
            used_indices.add(sample_id)
            selected_cases.append(
                {
                    "case_label": case_label,
                    "threshold": float(threshold),
                    "sample_index": sample_id,
                    "PatientGuid": row.get("PatientGuid"),
                    "y_true": int(row["y_true"]),
                    "y_pred": int(row["y_pred"]),
                    "y_score": float(row["y_score"]),
                    "error_type": row["error_type"],
                }
            )
            break
    return selected_cases


def compute_presence_stats(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "DMIndicator",
    min_support: int = 1,
) -> pd.DataFrame:
    base_rate = float(df[target_col].mean())
    rows: list[dict[str, Any]] = []
    for col in feature_cols:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            present = series.fillna(0) > 0
        else:
            present = series.notna() & series.astype(str).str.strip().ne("")

        support = int(present.sum())
        if support < min_support:
            continue

        dm_rate_when_present = float(df.loc[present, target_col].mean()) if support else float("nan")
        rows.append(
            {
                "feature": col,
                "support": support,
                "support_rate": float(support / len(df)),
                "dm_rate_when_present": dm_rate_when_present,
                "lift_vs_base": float(dm_rate_when_present / base_rate) if base_rate > 0 else float("nan"),
                "rate_diff_vs_base": float(dm_rate_when_present - base_rate),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["feature", "support", "support_rate", "dm_rate_when_present", "lift_vs_base", "rate_diff_vs_base"])

    return pd.DataFrame(rows).sort_values(["dm_rate_when_present", "support"], ascending=[False, False]).reset_index(drop=True)


def collect_keyword_hits(
    stats_df: pd.DataFrame,
    keyword_groups: dict[str, list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in stats_df.to_dict(orient="records"):
        feature = str(record["feature"])
        feature_key = feature.lower()
        matched_groups = [group_name for group_name, keywords in keyword_groups.items() if feature_key in set(keywords)]
        if not matched_groups:
            continue
        row = dict(record)
        row["matched_groups"] = ", ".join(matched_groups)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=list(stats_df.columns) + ["matched_groups"])
    return pd.DataFrame(rows).sort_values(["dm_rate_when_present", "support"], ascending=[False, False]).reset_index(drop=True)


def make_model_builders(neg_pos_ratio: float, random_state: int, use_class_balancing: bool = True) -> dict[str, Any]:
    return {
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=1000,
            solver="saga",
            penalty="l2",
            class_weight="balanced" if use_class_balancing else None,
            random_state=random_state,
        ),
        "SVM": lambda: SVC(
            kernel="linear",
            class_weight="balanced" if use_class_balancing else None,
            probability=False,
            random_state=random_state,
        ),
        "KNN": lambda: KNeighborsClassifier(
            n_neighbors=11,
            weights="distance",
            metric="minkowski",
            p=2,
            n_jobs=1,
        ),
        "Gradient Boosting": lambda: GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            random_state=random_state,
        ),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=neg_pos_ratio if use_class_balancing else 1.0,
            tree_method="hist",
            n_jobs=1,
            random_state=random_state,
        ),
        "LightGBM": lambda: LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced" if use_class_balancing else None,
            n_jobs=1,
            random_state=random_state,
            verbosity=-1,
        ),
    }


def evaluate_experiment(
    df: pd.DataFrame,
    feature_set_name: str,
    feature_cols: list[str],
    model_name: str,
    model_builder: Any,
    train_idx: pd.Index,
    test_idx: pd.Index,
    use_class_balancing: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    X = df[feature_cols].copy()
    y = df["DMIndicator"].astype(int)

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    preprocessor, role_info = build_preprocessor(df, feature_cols, model_name)
    model = model_builder()
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    fit_kwargs = get_model_fit_kwargs(model_name, y_train, use_class_balancing=use_class_balancing)
    pipeline.fit(X_train, y_train, **fit_kwargs)

    y_pred = pipeline.predict(X_test)
    y_score = get_positive_scores(pipeline, X_test)
    cm = confusion_matrix(y_test, y_pred)

    result = {
        "feature_set": feature_set_name,
        "model": model_name,
        "n_features": len(feature_cols),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "pr_auc": float(average_precision_score(y_test, y_score)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
    artifact = {
        "pipeline": pipeline,
        "role_info": role_info,
        "confusion_matrix": cm.tolist(),
    }
    return result, artifact


def fit_experiment_pipeline(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    model_builder: Any,
    train_idx: pd.Index,
    test_idx: pd.Index,
    use_class_balancing: bool = True,
) -> dict[str, Any]:
    X = df[feature_cols].copy()
    y = df["DMIndicator"].astype(int)

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    preprocessor, role_info = build_preprocessor(df, feature_cols, model_name)
    model = model_builder()
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    fit_kwargs = get_model_fit_kwargs(model_name, y_train, use_class_balancing=use_class_balancing)
    pipeline.fit(X_train, y_train, **fit_kwargs)
    y_score = get_positive_scores(pipeline, X_test)

    return {
        "pipeline": pipeline,
        "role_info": role_info,
        "X_test": X_test,
        "y_test": y_test,
        "y_score": y_score,
    }


def evaluate_experiment_with_optional_smote(
    df: pd.DataFrame,
    feature_set_name: str,
    feature_cols: list[str],
    model_name: str,
    model_builder: Any,
    train_idx: pd.Index,
    test_idx: pd.Index,
    use_smote: bool = False,
    random_state: int = 42,
    use_class_balancing: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    X = df[feature_cols].copy()
    y = df["DMIndicator"].astype(int)

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    preprocessor, role_info = build_preprocessor(df, feature_cols, model_name)
    X_train_prepared = preprocessor.fit_transform(X_train)
    X_test_prepared = preprocessor.transform(X_test)

    if use_smote:
        SMOTE = import_smote()
        smote = SMOTE(random_state=random_state)
        X_train_prepared, y_train = smote.fit_resample(X_train_prepared, y_train)

    model = model_builder()
    fit_kwargs = {}
    if not use_smote and use_class_balancing and model_name == "Gradient Boosting":
        fit_kwargs = {"sample_weight": compute_sample_weight(class_weight="balanced", y=y_train)}

    model.fit(X_train_prepared, y_train, **fit_kwargs)

    y_pred = model.predict(X_test_prepared)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_prepared)[:, 1]
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X_test_prepared)
        y_score = 1 / (1 + np.exp(-raw_scores))
    else:
        raise AttributeError("Model does not provide predict_proba or decision_function.")

    cm = confusion_matrix(y_test, y_pred)
    strategy = "smote" if use_smote else "no_smote"
    result = {
        "feature_set": feature_set_name,
        "model": model_name,
        "sampling_strategy": strategy,
        "n_features": len(feature_cols),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "pr_auc": float(average_precision_score(y_test, y_score)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
    artifact = {
        "preprocessor": preprocessor,
        "model": model,
        "role_info": role_info,
        "confusion_matrix": cm.tolist(),
    }
    return result, artifact


def compute_threshold_metrics(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "predicted_positive_rate": float(y_pred.mean()),
    }


def label_error_types(y_true: pd.Series, y_pred: np.ndarray) -> np.ndarray:
    y_true_array = np.asarray(y_true).astype(int)
    y_pred_array = np.asarray(y_pred).astype(int)
    labels = np.empty(len(y_true_array), dtype=object)
    labels[(y_true_array == 1) & (y_pred_array == 1)] = "TP"
    labels[(y_true_array == 0) & (y_pred_array == 0)] = "TN"
    labels[(y_true_array == 0) & (y_pred_array == 1)] = "FP"
    labels[(y_true_array == 1) & (y_pred_array == 0)] = "FN"
    return labels


def get_interpretable_context_columns(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    preferred_cols = [
        "PatientGuid",
        "Gender",
        "State",
        "Age",
        "Height_Min",
        "Height_Max",
        "Height_Mean",
        "Weight_Min",
        "Weight_Max",
        "Weight_Mean",
        "BMI_Min",
        "BMI_Max",
        "BMI_Mean",
        "SystolicBP_Min",
        "SystolicBP_Max",
        "SystolicBP_Mean",
        "DiastolicBP_Min",
        "DiastolicBP_Max",
        "DiastolicBP_Mean",
        "RespiratoryRate_Min",
        "RespiratoryRate_Max",
        "RespiratoryRate_Mean",
        "Temperature_Min",
        "Temperature_Max",
        "Temperature_Mean",
        "DiagnosisCount",
        "VisitCount",
        "DiagnosisFreq",
        "AcuteCount",
        "AcuteFreq",
    ]
    available_cols = [col for col in preferred_cols if col in df.columns and (col == "PatientGuid" or col in feature_cols)]
    return list(dict.fromkeys(available_cols))


def build_error_numeric_profile(
    test_frame: pd.DataFrame,
    threshold: float,
    numeric_cols: list[str],
) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame(columns=["threshold", "error_type", "feature", "mean", "median", "missing_rate"])

    rows: list[dict[str, Any]] = []
    for error_type, group in test_frame.groupby("error_type"):
        for feature in numeric_cols:
            series = pd.to_numeric(group[feature], errors="coerce")
            rows.append(
                {
                    "threshold": float(threshold),
                    "error_type": error_type,
                    "feature": feature,
                    "mean": float(series.mean()) if not series.empty else float("nan"),
                    "median": float(series.median()) if not series.empty else float("nan"),
                    "missing_rate": float(series.isna().mean()) if len(series) > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def build_error_binary_pattern_summary(
    test_frame: pd.DataFrame,
    binary_cols: list[str],
    threshold: float,
    top_n: int = 20,
    min_present_support: int = 10,
) -> pd.DataFrame:
    if not binary_cols:
        return pd.DataFrame(
            columns=[
                "threshold",
                "comparison_group",
                "feature",
                "error_rate",
                "reference_rate",
                "delta_error_minus_reference",
                "support_error_group",
                "support_reference_group",
                "present_support",
            ]
        )

    rows: list[dict[str, Any]] = []
    comparisons = [
        ("FN_vs_TP", "FN", "TP"),
        ("FP_vs_TN", "FP", "TN"),
    ]
    for comparison_group, error_label, reference_label in comparisons:
        error_group = test_frame[test_frame["error_type"] == error_label]
        reference_group = test_frame[test_frame["error_type"] == reference_label]
        if error_group.empty or reference_group.empty:
            continue

        for feature in binary_cols:
            error_series = pd.to_numeric(error_group[feature], errors="coerce").fillna(0)
            reference_series = pd.to_numeric(reference_group[feature], errors="coerce").fillna(0)
            present_support = int((error_series > 0).sum() + (reference_series > 0).sum())
            if present_support < min_present_support:
                continue

            error_rate = float(error_series.mean())
            reference_rate = float(reference_series.mean())
            delta = error_rate - reference_rate
            if np.isclose(error_rate, 0.0) and np.isclose(reference_rate, 0.0):
                continue
            rows.append(
                {
                    "threshold": float(threshold),
                    "comparison_group": comparison_group,
                    "feature": feature,
                    "error_rate": error_rate,
                    "reference_rate": reference_rate,
                    "delta_error_minus_reference": delta,
                    "support_error_group": int(len(error_group)),
                    "support_reference_group": int(len(reference_group)),
                    "present_support": present_support,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "threshold",
                "comparison_group",
                "feature",
                "error_rate",
                "reference_rate",
                "delta_error_minus_reference",
                "support_error_group",
                "support_reference_group",
                "present_support",
            ]
        )

    pattern_df = pd.DataFrame(rows)
    top_frames: list[pd.DataFrame] = []
    for comparison_group, group_df in pattern_df.groupby("comparison_group"):
        top_group = (
            group_df.assign(abs_delta=group_df["delta_error_minus_reference"].abs())
            .sort_values(["abs_delta", "error_rate"], ascending=[False, False])
            .head(top_n)
            .drop(columns="abs_delta")
        )
        top_frames.append(top_group)
    return pd.concat(top_frames, ignore_index=True)


def summarize_error_thresholds(
    threshold_rows: list[dict[str, Any]],
    prediction_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    summary_df = pd.DataFrame(threshold_rows)
    if summary_df.empty:
        return summary_df

    prediction_df = pd.concat(prediction_frames, ignore_index=True)
    count_df = (
        prediction_df.groupby(["threshold", "error_type"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    count_pivot = count_df.pivot(index="threshold", columns="error_type", values="count").fillna(0).reset_index()
    count_pivot.columns.name = None
    rename_map = {"TP": "tp_count", "TN": "tn_count", "FP": "fp_count", "FN": "fn_count"}
    for label in rename_map:
        if label not in count_pivot.columns:
            count_pivot[label] = 0
    count_pivot = count_pivot.rename(columns=rename_map)
    merged = summary_df.merge(count_pivot, on="threshold", how="left")
    for renamed_label in rename_map.values():
        merged[renamed_label] = merged[renamed_label].fillna(0).astype(int)
    return merged.sort_values("threshold").reset_index(drop=True)


def run_error_analysis(
    dataset_path: Path = APP_FINAL_DATASET_PATH,
    out_dir: Path = TRAIN_OUTPUT_DIR,
    feature_set_name: str = DEFAULT_ERROR_ANALYSIS_FEATURE_SET,
    model_name: str = DEFAULT_ERROR_ANALYSIS_MODEL,
    thresholds: list[float] | None = None,
    test_size: float = 0.3,
    random_state: int = 42,
    use_class_balancing: bool = True,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    df = load_csv(dataset_path)
    feature_sets, _ = build_feature_sets(df)

    if feature_set_name not in feature_sets:
        raise ValueError(f"Feature set `{feature_set_name}` tidak ditemukan.")

    thresholds_to_use = sorted(set(thresholds or DEFAULT_ERROR_ANALYSIS_THRESHOLDS))
    if not thresholds_to_use:
        raise ValueError("Minimal harus ada satu threshold untuk error analysis.")

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        stratify=df["DMIndicator"],
        random_state=random_state,
    )
    y_train_ref = df.loc[train_idx, "DMIndicator"].astype(int)
    neg_pos_ratio = float((y_train_ref == 0).sum() / (y_train_ref == 1).sum())
    model_builders = make_model_builders(
        neg_pos_ratio=neg_pos_ratio,
        random_state=random_state,
        use_class_balancing=use_class_balancing,
    )
    if model_name not in model_builders:
        raise ValueError(f"Model `{model_name}` tidak ditemukan.")

    feature_cols = feature_sets[feature_set_name]
    fit_result = fit_experiment_pipeline(
        df=df,
        feature_cols=feature_cols,
        model_name=model_name,
        model_builder=model_builders[model_name],
        train_idx=train_idx,
        test_idx=test_idx,
        use_class_balancing=use_class_balancing,
    )

    base_test_frame = df.loc[test_idx, get_interpretable_context_columns(df, feature_cols)].copy()
    if "PatientGuid" not in base_test_frame.columns and "PatientGuid" in df.columns:
        base_test_frame.insert(0, "PatientGuid", df.loc[test_idx, "PatientGuid"].values)
    base_test_frame["y_true"] = fit_result["y_test"].values
    base_test_frame["y_score"] = fit_result["y_score"]

    context_numeric_cols = [
        col for col in base_test_frame.columns if col not in {"PatientGuid", "Gender", "State", "y_true", "y_score"} and pd.api.types.is_numeric_dtype(base_test_frame[col])
    ]
    _, binary_like_cols, _ = split_feature_roles(df, feature_cols)

    threshold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    numeric_profile_frames: list[pd.DataFrame] = []
    binary_pattern_frames: list[pd.DataFrame] = []

    strategy = describe_feature_set(feature_set_name)
    for threshold in thresholds_to_use:
        metrics = compute_threshold_metrics(
            y_true=fit_result["y_test"],
            y_score=fit_result["y_score"],
            threshold=threshold,
        )
        metrics["feature_set"] = feature_set_name
        metrics["feature_set_label"] = strategy["label"]
        metrics["feature_set_role"] = strategy["role"]
        metrics["model"] = model_name
        metrics["use_class_balancing"] = use_class_balancing
        threshold_rows.append(metrics)

        threshold_frame = base_test_frame.copy()
        y_pred = (fit_result["y_score"] >= threshold).astype(int)
        threshold_frame["threshold"] = float(threshold)
        threshold_frame["y_pred"] = y_pred
        threshold_frame["error_type"] = label_error_types(threshold_frame["y_true"], y_pred)
        threshold_frame["is_correct"] = (threshold_frame["y_true"] == threshold_frame["y_pred"]).astype(int)
        threshold_frame["score_gap_from_threshold"] = threshold_frame["y_score"] - threshold
        prediction_frames.append(threshold_frame)

        numeric_profile_frames.append(
            build_error_numeric_profile(
                test_frame=threshold_frame,
                threshold=threshold,
                numeric_cols=context_numeric_cols,
            )
        )
        binary_pattern_frames.append(
            build_error_binary_pattern_summary(
                test_frame=df.loc[test_idx, feature_cols].copy().assign(error_type=threshold_frame["error_type"].values),
                binary_cols=binary_like_cols,
                threshold=threshold,
            )
        )

    summary_df = summarize_error_thresholds(threshold_rows, prediction_frames)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    numeric_profile_df = pd.concat(numeric_profile_frames, ignore_index=True) if numeric_profile_frames else pd.DataFrame()
    binary_pattern_df = pd.concat(binary_pattern_frames, ignore_index=True) if binary_pattern_frames else pd.DataFrame()

    summary_path = out_dir / "error_analysis_summary.csv"
    predictions_path = out_dir / "error_analysis_predictions.csv"
    numeric_profile_path = out_dir / "error_analysis_numeric_profile.csv"
    binary_pattern_path = out_dir / "error_analysis_binary_patterns.csv"

    summary_df.to_csv(summary_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    numeric_profile_df.to_csv(numeric_profile_path, index=False)
    binary_pattern_df.to_csv(binary_pattern_path, index=False)

    lines: list[str] = []
    lines.append("# Error Analysis Report")
    lines.append("")
    lines.append(f"- Dataset: `{dataset_path}`")
    lines.append(f"- Feature set: `{feature_set_name}` ({strategy['label']})")
    lines.append(f"- Model: `{model_name}`")
    lines.append(f"- Test size: `{test_size}`")
    lines.append(f"- Random state: `{random_state}`")
    lines.append(f"- Weighted model: `{use_class_balancing}`")
    lines.append(f"- Thresholds: `{', '.join(f'{value:.2f}' for value in thresholds_to_use)}`")
    lines.append("")
    lines.append("## Ringkasan Threshold")
    lines.append("")
    for row in summary_df.to_dict(orient="records"):
        lines.append(
            f"- Threshold `{row['threshold']:.2f}`: TP=`{row['tp_count']}`, TN=`{row['tn_count']}`, FP=`{row['fp_count']}`, FN=`{row['fn_count']}`, "
            f"Recall=`{row['recall']:.4f}`, Precision=`{row['precision']:.4f}`, F1=`{row['f1']:.4f}`"
        )
    lines.append("")
    lines.append("## Pola Error yang Perlu Diperhatikan")
    lines.append("")
    if len(summary_df) >= 2:
        low_row = summary_df.iloc[0]
        high_row = summary_df.iloc[-1]
        lines.append(
            f"- Saat threshold dinaikkan dari `{low_row['threshold']:.2f}` ke `{high_row['threshold']:.2f}`, "
            f"FP turun dari `{int(low_row['fp_count'])}` menjadi `{int(high_row['fp_count'])}`, "
            f"tetapi FN naik dari `{int(low_row['fn_count'])}` menjadi `{int(high_row['fn_count'])}`."
        )

    focus_features = [feature for feature in ["Age", "BMI_Mean", "Weight_Mean", "DiagnosisCount", "VisitCount"] if feature in context_numeric_cols]
    for threshold in thresholds_to_use:
        threshold_numeric = numeric_profile_df[numeric_profile_df["threshold"] == float(threshold)]
        for error_label, reference_label, description in [
            ("FN", "TP", "pasien DM yang terlewat"),
            ("FP", "TN", "pasien non-DM yang salah terdeteksi"),
        ]:
            diff_rows: list[tuple[str, float, float, float]] = []
            for feature in focus_features:
                error_row = threshold_numeric[
                    (threshold_numeric["error_type"] == error_label) & (threshold_numeric["feature"] == feature)
                ]
                reference_row = threshold_numeric[
                    (threshold_numeric["error_type"] == reference_label) & (threshold_numeric["feature"] == feature)
                ]
                if error_row.empty or reference_row.empty:
                    continue
                error_mean = float(error_row.iloc[0]["mean"])
                reference_mean = float(reference_row.iloc[0]["mean"])
                diff_rows.append((feature, error_mean, reference_mean, error_mean - reference_mean))
            if diff_rows:
                top_diffs = sorted(diff_rows, key=lambda item: abs(item[3]), reverse=True)[:3]
                diff_text = "; ".join(
                    f"`{feature}` `{error_mean:.2f}` vs `{reference_mean:.2f}` (delta `{delta:+.2f}`)"
                    for feature, error_mean, reference_mean, delta in top_diffs
                )
                lines.append(f"- Threshold `{threshold:.2f}` | `{description}`: {diff_text}")

    if not binary_pattern_df.empty:
        lines.append("")
        lines.append("## Fitur Biner yang Paling Membeda")
        lines.append("")
        for threshold in thresholds_to_use:
            threshold_patterns = binary_pattern_df[binary_pattern_df["threshold"] == float(threshold)]
            for comparison_group in ["FN_vs_TP", "FP_vs_TN"]:
                top_pattern = threshold_patterns[threshold_patterns["comparison_group"] == comparison_group].head(3)
                if top_pattern.empty:
                    continue
                pattern_text = "; ".join(
                    f"`{row['feature']}` (delta `{row['delta_error_minus_reference']:.3f}`, support `{int(row['present_support'])}`)"
                    for row in top_pattern.to_dict(orient="records")
                )
                lines.append(f"- Threshold `{threshold:.2f}` | `{comparison_group}`: {pattern_text}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Summary CSV: `{summary_path}`")
    lines.append(f"- Predictions CSV: `{predictions_path}`")
    lines.append(f"- Numeric profile CSV: `{numeric_profile_path}`")
    lines.append(f"- Binary pattern CSV: `{binary_pattern_path}`")
    report_path = out_dir / "error_analysis_report.md"
    save_markdown(report_path, "\n".join(lines))

    return {
        "dataset_path": dataset_path,
        "out_dir": out_dir,
        "summary_path": summary_path,
        "predictions_path": predictions_path,
        "numeric_profile_path": numeric_profile_path,
        "binary_pattern_path": binary_pattern_path,
        "report_path": report_path,
        "summary_df": summary_df,
        "predictions_df": predictions_df,
        "numeric_profile_df": numeric_profile_df,
        "binary_pattern_df": binary_pattern_df,
    }


def run_xai_analysis(
    dataset_path: Path = APP_FINAL_DATASET_PATH,
    out_dir: Path = TRAIN_OUTPUT_DIR,
    feature_set_name: str = DEFAULT_XAI_FEATURE_SET,
    model_name: str = DEFAULT_XAI_MODEL,
    threshold: float = DEFAULT_XAI_THRESHOLD,
    test_size: float = 0.3,
    random_state: int = 42,
    use_class_balancing: bool = True,
    max_display: int = 20,
    max_dependence_plots: int = 3,
) -> dict[str, Any]:
    shap = import_shap()
    xai_dir = ensure_dir(out_dir / "xai")

    df = load_csv(dataset_path)
    feature_sets, _ = build_feature_sets(df)
    if feature_set_name not in feature_sets:
        raise ValueError(f"Feature set `{feature_set_name}` tidak ditemukan.")

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        stratify=df["DMIndicator"],
        random_state=random_state,
    )
    y_train_ref = df.loc[train_idx, "DMIndicator"].astype(int)
    neg_pos_ratio = float((y_train_ref == 0).sum() / (y_train_ref == 1).sum())
    model_builders = make_model_builders(
        neg_pos_ratio=neg_pos_ratio,
        random_state=random_state,
        use_class_balancing=use_class_balancing,
    )
    if model_name not in model_builders:
        raise ValueError(f"Model `{model_name}` tidak ditemukan.")

    feature_cols = feature_sets[feature_set_name]
    fit_result = fit_experiment_pipeline(
        df=df,
        feature_cols=feature_cols,
        model_name=model_name,
        model_builder=model_builders[model_name],
        train_idx=train_idx,
        test_idx=test_idx,
        use_class_balancing=use_class_balancing,
    )

    pipeline = fit_result["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    X_train = df.loc[train_idx, feature_cols].copy()
    X_test = fit_result["X_test"].copy()
    y_test = fit_result["y_test"].copy()
    y_score = np.asarray(fit_result["y_score"])
    y_pred = (y_score >= threshold).astype(int)

    X_train_prepared = preprocessor.transform(X_train)
    X_test_prepared = preprocessor.transform(X_test)
    feature_names = get_preprocessed_feature_names(preprocessor)
    X_test_prepared_df = pd.DataFrame(X_test_prepared, columns=feature_names, index=X_test.index)

    tree_model_names = {"Gradient Boosting", "XGBoost", "LightGBM"}
    if model_name in tree_model_names:
        shap_values, base_value = compute_tree_shap_values(
            shap=shap,
            model_name=model_name,
            model=model,
            X_test_prepared=X_test_prepared,
            feature_names=feature_names,
        )
    else:
        background_size = min(200, len(X_train_prepared))
        background = X_train_prepared[:background_size]
        explainer = shap.Explainer(model, background)
        raw_shap_output = explainer(X_test_prepared)
        shap_values, base_value = normalize_shap_output(raw_shap_output)
    if shap_values.shape[1] != X_test_prepared_df.shape[1]:
        raise ValueError("Jumlah kolom SHAP tidak sama dengan jumlah fitur hasil preprocessing.")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_feature_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs_shap,
                "mean_feature_value": X_test_prepared_df.mean(axis=0).values,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    top_feature_path = xai_dir / "xai_top_features.csv"
    top_feature_df.to_csv(top_feature_path, index=False)

    plt.figure()
    shap.summary_plot(shap_values, X_test_prepared_df, show=False, max_display=max_display)
    plt.tight_layout()
    summary_plot_path = xai_dir / "shap_summary_plot.png"
    plt.savefig(summary_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_test_prepared_df, plot_type="bar", show=False, max_display=max_display)
    plt.tight_layout()
    summary_bar_path = xai_dir / "shap_summary_bar.png"
    plt.savefig(summary_bar_path, dpi=200, bbox_inches="tight")
    plt.close()

    dependence_paths: list[Path] = []
    dependence_features = top_feature_df["feature"].head(max_dependence_plots).tolist()
    for feature_name in dependence_features:
        plt.figure()
        shap.dependence_plot(feature_name, shap_values, X_test_prepared_df, interaction_index=None, show=False)
        plt.tight_layout()
        dependence_path = xai_dir / f"shap_dependence_{feature_name.replace('/', '_').replace(' ', '_')}.png"
        plt.savefig(dependence_path, dpi=200, bbox_inches="tight")
        plt.close()
        dependence_paths.append(dependence_path)

    context_cols = get_interpretable_context_columns(df, feature_cols)
    prediction_context = df.loc[test_idx, context_cols].copy()
    if "PatientGuid" not in prediction_context.columns and "PatientGuid" in df.columns:
        prediction_context.insert(0, "PatientGuid", df.loc[test_idx, "PatientGuid"].values)
    prediction_context["sample_index"] = prediction_context.index
    prediction_context["y_true"] = y_test.values
    prediction_context["y_score"] = y_score
    prediction_context["y_pred"] = y_pred
    prediction_context["error_type"] = label_error_types(y_test, y_pred)
    prediction_context["threshold"] = float(threshold)

    representative_cases = select_representative_cases(prediction_context, threshold=threshold)
    local_case_rows: list[dict[str, Any]] = []
    local_case_paths: list[dict[str, str]] = []
    position_lookup = {index_value: position for position, index_value in enumerate(X_test.index)}

    for case in representative_cases:
        sample_index = case["sample_index"]
        sample_position = position_lookup[sample_index]
        sample_series = X_test_prepared_df.iloc[sample_position]
        sample_values = shap_values[sample_position]

        local_importance_df = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "feature_value": sample_series.values,
                    "shap_value": sample_values,
                    "abs_shap_value": np.abs(sample_values),
                }
            )
            .sort_values("abs_shap_value", ascending=False)
            .reset_index(drop=True)
        )
        local_csv_path = xai_dir / f"xai_local_{case['case_label']}.csv"
        local_importance_df.to_csv(local_csv_path, index=False)

        force_plot = shap.force_plot(
            base_value,
            sample_values,
            sample_series.values,
            feature_names=feature_names,
            matplotlib=False,
        )
        force_plot_path = xai_dir / f"xai_force_{case['case_label']}.html"
        shap.save_html(str(force_plot_path), force_plot)

        local_case_rows.append(
            {
                **case,
                "top_features": ", ".join(local_importance_df["feature"].head(5).tolist()),
            }
        )
        local_case_paths.append(
            {
                "case_label": case["case_label"],
                "csv_path": str(local_csv_path),
                "force_plot_path": str(force_plot_path),
            }
        )

    local_cases_df = pd.DataFrame(local_case_rows)
    local_cases_path = xai_dir / "xai_local_cases_summary.csv"
    local_cases_df.to_csv(local_cases_path, index=False)

    strategy = describe_feature_set(feature_set_name)
    lines: list[str] = []
    lines.append("# XAI Report")
    lines.append("")
    lines.append(f"- Dataset: `{dataset_path}`")
    lines.append(f"- Feature set: `{feature_set_name}` ({strategy['label']})")
    lines.append(f"- Model: `{model_name}`")
    lines.append(f"- Threshold klasifikasi: `{threshold:.2f}`")
    lines.append(f"- Test size: `{test_size}`")
    lines.append(f"- Random state: `{random_state}`")
    lines.append(f"- Weighted model: `{use_class_balancing}`")
    lines.append("")
    lines.append("## Ringkasan Sesuai Proposal")
    lines.append("")
    lines.append("- SHAP digunakan pada model terbaik untuk membaca kontribusi fitur secara global dan lokal.")
    lines.append("- Output yang disiapkan mengikuti arah proposal: `summary plot`, `dependence plot`, dan `force plot` untuk pasien individual.")
    lines.append("")
    lines.append("## Fitur Paling Berpengaruh")
    lines.append("")
    for row in top_feature_df.head(10).to_dict(orient="records"):
        lines.append(f"- `{row['feature']}`: mean |SHAP| = `{row['mean_abs_shap']:.4f}`")
    lines.append("")
    lines.append("## Contoh Pasien Individual")
    lines.append("")
    if local_case_rows:
        for case in local_case_rows:
            lines.append(
                f"- `{case['case_label']}` | error_type=`{case['error_type']}` | "
                f"score=`{case['y_score']:.4f}` | top features: {case['top_features']}"
            )
    else:
        lines.append("- Tidak ada sampel lokal yang berhasil dipilih.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Summary plot: `{summary_plot_path}`")
    lines.append(f"- Summary bar plot: `{summary_bar_path}`")
    for dependence_path in dependence_paths:
        lines.append(f"- Dependence plot: `{dependence_path}`")
    lines.append(f"- Top feature CSV: `{top_feature_path}`")
    lines.append(f"- Local case summary CSV: `{local_cases_path}`")
    for case_path in local_case_paths:
        lines.append(f"- Local case `{case_path['case_label']}` CSV: `{case_path['csv_path']}`")
        lines.append(f"- Local case `{case_path['case_label']}` force plot: `{case_path['force_plot_path']}`")
    report_path = xai_dir / "xai_report.md"
    save_markdown(report_path, "\n".join(lines))

    return {
        "dataset_path": dataset_path,
        "out_dir": xai_dir,
        "report_path": report_path,
        "summary_plot_path": summary_plot_path,
        "summary_bar_path": summary_bar_path,
        "dependence_paths": dependence_paths,
        "top_feature_path": top_feature_path,
        "local_cases_path": local_cases_path,
        "feature_importance_df": top_feature_df,
        "local_cases_df": local_cases_df,
    }


def summarize_cv_results(results_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        results_df.groupby(["feature_set", "feature_set_label", "feature_set_role", "model", "n_features"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_precision=("precision", "mean"),
            std_precision=("precision", "std"),
            mean_recall=("recall", "mean"),
            std_recall=("recall", "std"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            total_tn=("tn", "sum"),
            total_fp=("fp", "sum"),
            total_fn=("fn", "sum"),
            total_tp=("tp", "sum"),
        )
        .sort_values(["mean_pr_auc", "mean_recall", "mean_f1"], ascending=False)
        .reset_index(drop=True)
    )

    std_cols = [col for col in summary_df.columns if col.startswith("std_")]
    if std_cols:
        summary_df[std_cols] = summary_df[std_cols].fillna(0.0)
    return summary_df


def run_main_cv(
    dataset_path: Path = APP_FINAL_DATASET_PATH,
    out_dir: Path = TRAIN_OUTPUT_DIR,
    selected_feature_sets: list[str] | None = None,
    selected_models: list[str] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    use_class_balancing: bool = True,
) -> dict[str, Any]:
    if n_splits < 2:
        raise ValueError("n_splits minimal 2 untuk Stratified K-Fold.")

    ensure_dir(out_dir)
    df = load_csv(dataset_path)
    feature_sets, _ = build_feature_sets(df)

    if selected_feature_sets is None:
        feature_set_names = [name for name in DEFAULT_CV_FEATURE_SETS if name in feature_sets]
    else:
        feature_set_names = [name for name in selected_feature_sets if name in feature_sets]

    feature_sets = {name: feature_sets[name] for name in feature_set_names}
    if not feature_sets:
        raise ValueError("Tidak ada feature set utama yang tersedia untuk CV.")

    y = df["DMIndicator"].astype(int)
    neg_count = int((y == 0).sum())
    pos_count = int((y == 1).sum())
    if n_splits > min(neg_count, pos_count):
        raise ValueError(
            f"n_splits={n_splits} terlalu besar untuk distribusi kelas saat ini. "
            f"Maksimal aman adalah {min(neg_count, pos_count)}."
        )

    initial_neg_pos_ratio = float(neg_count / pos_count)
    model_builders = make_model_builders(
        neg_pos_ratio=initial_neg_pos_ratio,
        random_state=random_state,
        use_class_balancing=use_class_balancing,
    )
    if selected_models:
        model_builders = {name: builder for name, builder in model_builders.items() if name in selected_models}
    if not model_builders:
        raise ValueError("Tidak ada model yang tersedia untuk CV.")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    index_values = df.index.to_numpy()
    fold_results: list[dict[str, Any]] = []

    for fold_number, (train_positions, test_positions) in enumerate(splitter.split(df, y), start=1):
        train_idx = index_values[train_positions]
        test_idx = index_values[test_positions]
        y_train_ref = df.loc[train_idx, "DMIndicator"].astype(int)
        neg_pos_ratio = float((y_train_ref == 0).sum() / (y_train_ref == 1).sum())
        fold_model_builders = make_model_builders(
            neg_pos_ratio=neg_pos_ratio,
            random_state=random_state,
            use_class_balancing=use_class_balancing,
        )

        for feature_set_name, feature_cols in feature_sets.items():
            strategy = describe_feature_set(feature_set_name)
            for model_name in model_builders:
                result, _ = evaluate_experiment(
                    df=df,
                    feature_set_name=feature_set_name,
                    feature_cols=feature_cols,
                    model_name=model_name,
                    model_builder=fold_model_builders[model_name],
                    train_idx=train_idx,
                    test_idx=test_idx,
                    use_class_balancing=use_class_balancing,
                )
                result["fold"] = fold_number
                result["feature_set_label"] = strategy["label"]
                result["feature_set_role"] = strategy["role"]
                result["use_class_balancing"] = use_class_balancing
                result["experiment_key"] = f"{model_name}__{feature_set_name}__fold_{fold_number}"
                fold_results.append(result)

    fold_results_df = pd.DataFrame(fold_results).sort_values(
        ["feature_set", "model", "fold"], ascending=[True, True, True]
    ).reset_index(drop=True)
    fold_results_path = out_dir / "main_cv_fold_results.csv"
    fold_results_df.to_csv(fold_results_path, index=False)

    summary_df = summarize_cv_results(fold_results_df)
    summary_path = out_dir / "main_cv_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    best_row = summary_df.iloc[0].to_dict()
    lines: list[str] = []
    lines.append("# Main Cross-Validation Report")
    lines.append("")
    lines.append(f"- Dataset: `{dataset_path}`")
    lines.append(f"- CV strategy: `StratifiedKFold(n_splits={n_splits}, shuffle=True, random_state={random_state})`")
    lines.append(f"- Weighted model: `{use_class_balancing}`")
    lines.append("")
    lines.append("## Tujuan")
    lines.append("")
    lines.append("- Memberikan evaluasi yang lebih stabil untuk perbandingan model utama.")
    lines.append("- CV ini hanya memakai feature set primary agar tetap selaras dengan tujuan prediksi sebelum medication diberikan.")
    lines.append("")
    lines.append("## Feature Set yang Diuji")
    lines.append("")
    for feature_set_name in feature_sets:
        strategy = describe_feature_set(feature_set_name)
        lines.append(
            f"- `{feature_set_name}` ({strategy['label']} | role=`{strategy['role']}`): {strategy['description']}"
        )
    lines.append("")
    lines.append("## Ringkasan Hasil")
    lines.append("")
    for row in summary_df.to_dict(orient="records"):
        lines.append(
            f"- `{row['feature_set']}` + `{row['model']}`: "
            f"PR-AUC `{row['mean_pr_auc']:.4f} +/- {row['std_pr_auc']:.4f}`, "
            f"Recall `{row['mean_recall']:.4f} +/- {row['std_recall']:.4f}`, "
            f"Precision `{row['mean_precision']:.4f} +/- {row['std_precision']:.4f}`, "
            f"F1 `{row['mean_f1']:.4f} +/- {row['std_f1']:.4f}`, "
            f"ROC-AUC `{row['mean_roc_auc']:.4f} +/- {row['std_roc_auc']:.4f}`"
        )
    lines.append("")
    lines.append("## Model CV Terbaik")
    lines.append("")
    lines.append(f"- Feature set: `{best_row['feature_set']}`")
    lines.append(f"- Model: `{best_row['model']}`")
    lines.append(f"- Mean PR-AUC: `{best_row['mean_pr_auc']:.4f}`")
    lines.append(f"- Mean Recall: `{best_row['mean_recall']:.4f}`")
    lines.append(f"- Mean Precision: `{best_row['mean_precision']:.4f}`")
    lines.append(f"- Mean F1: `{best_row['mean_f1']:.4f}`")
    lines.append("")
    lines.append("## Catatan")
    lines.append("")
    lines.append("- Nilai mean/std membantu melihat apakah performa model stabil di beberapa fold, bukan hanya kebetulan satu split.")
    lines.append("- KNN tetap ikut dalam CV utama sebagai pembanding metode berbasis jarak.")
    lines.append("- Feature set `full` tidak dimasukkan ke CV utama karena masih mengandung medication dan lebih cocok diposisikan sebagai comparator leakage.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Fold-level results CSV: `{fold_results_path}`")
    lines.append(f"- Summary CSV: `{summary_path}`")
    report_path = out_dir / "main_cv_report.md"
    save_markdown(report_path, "\n".join(lines))

    return {
        "dataset_path": dataset_path,
        "out_dir": out_dir,
        "results_path": fold_results_path,
        "summary_path": summary_path,
        "report_path": report_path,
        "fold_results_df": fold_results_df,
        "summary_df": summary_df,
        "best_experiment": best_row,
    }


def run_training(
    dataset_path: Path = APP_FINAL_DATASET_PATH,
    out_dir: Path = TRAIN_OUTPUT_DIR,
    selected_feature_sets: list[str] | None = None,
    selected_models: list[str] | None = None,
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    df = load_csv(dataset_path)
    feature_sets, metadata = build_feature_sets(df)

    if selected_feature_sets:
        feature_sets = {name: cols for name, cols in feature_sets.items() if name in selected_feature_sets}

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        stratify=df["DMIndicator"],
        random_state=random_state,
    )
    y_train_ref = df.loc[train_idx, "DMIndicator"].astype(int)
    neg_pos_ratio = float((y_train_ref == 0).sum() / (y_train_ref == 1).sum())

    model_builders = make_model_builders(neg_pos_ratio=neg_pos_ratio, random_state=random_state)
    if selected_models:
        model_builders = {name: builder for name, builder in model_builders.items() if name in selected_models}

    results: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    for feature_set_name, feature_cols in feature_sets.items():
        for model_name, model_builder in model_builders.items():
            result, artifact = evaluate_experiment(df, feature_set_name, feature_cols, model_name, model_builder, train_idx, test_idx)
            experiment_key = f"{model_name}__{feature_set_name}"
            result["experiment_key"] = experiment_key
            strategy = describe_feature_set(feature_set_name)
            result["feature_set_label"] = strategy["label"]
            result["feature_set_role"] = strategy["role"]
            results.append(result)
            artifacts[experiment_key] = artifact

    results_df = pd.DataFrame(results).sort_values(["pr_auc", "recall", "f1"], ascending=False).reset_index(drop=True)
    best_overall_row = results_df.iloc[0].to_dict()
    primary_results_df = results_df[results_df["feature_set_role"] == "primary"].reset_index(drop=True)
    if primary_results_df.empty:
        raise ValueError("Tidak ada feature set primary yang tersedia. Minimal harus ada satu feature set tanpa medication.")

    best_primary_row = primary_results_df.iloc[0].to_dict()
    best_primary_key = best_primary_row["experiment_key"]
    comparator_results_df = results_df[results_df["feature_set_role"] == "comparator"].reset_index(drop=True)
    best_comparator_row = comparator_results_df.iloc[0].to_dict() if not comparator_results_df.empty else None

    results_df.to_csv(out_dir / "baseline_results.csv", index=False)
    save_json(out_dir / "feature_metadata.json", metadata)
    serializable_artifacts = {
        key: {artifact_key: artifact_value for artifact_key, artifact_value in artifact.items() if artifact_key != "pipeline"}
        for key, artifact in artifacts.items()
    }
    save_json(out_dir / "artifacts.json", serializable_artifacts)

    best_pipeline = artifacts[best_primary_key]["pipeline"]
    with (out_dir / "best_model.pkl").open("wb") as handle:
        pickle.dump(best_pipeline, handle)

    if best_comparator_row is not None:
        comparator_pipeline = artifacts[best_comparator_row["experiment_key"]]["pipeline"]
        with (out_dir / "best_comparator_model.pkl").open("wb") as handle:
            pickle.dump(comparator_pipeline, handle)

    lines: list[str] = []
    lines.append("# Training Report")
    lines.append("")
    lines.append(f"- Dataset: `{dataset_path}`")
    lines.append(f"- Test size: `{test_size}`")
    lines.append(f"- Random state: `{random_state}`")
    lines.append("")
    lines.append("## Struktur Feature Set")
    lines.append("")
    for feature_set_name in feature_sets:
        strategy = describe_feature_set(feature_set_name)
        lines.append(
            f"- `{feature_set_name}` ({strategy['label']} | role=`{strategy['role']}`): {strategy['description']}"
        )
    lines.append("")
    lines.append("## Best Primary Model")
    lines.append("")
    lines.append(f"- Model: `{best_primary_row['model']}`")
    lines.append(f"- Feature set: `{best_primary_row['feature_set']}`")
    lines.append(f"- Label: `{best_primary_row['feature_set_label']}`")
    lines.append(f"- PR-AUC: `{best_primary_row['pr_auc']:.4f}`")
    lines.append(f"- ROC-AUC: `{best_primary_row['roc_auc']:.4f}`")
    lines.append(f"- Recall: `{best_primary_row['recall']:.4f}`")
    lines.append(f"- Precision: `{best_primary_row['precision']:.4f}`")
    lines.append(f"- F1-score: `{best_primary_row['f1']:.4f}`")
    lines.append("")
    lines.append("## Best Overall Experiment")
    lines.append("")
    lines.append(f"- Model: `{best_overall_row['model']}`")
    lines.append(f"- Feature set: `{best_overall_row['feature_set']}`")
    lines.append(f"- Label: `{best_overall_row['feature_set_label']}`")
    lines.append(f"- Role: `{best_overall_row['feature_set_role']}`")
    lines.append(f"- PR-AUC: `{best_overall_row['pr_auc']:.4f}`")
    lines.append(f"- ROC-AUC: `{best_overall_row['roc_auc']:.4f}`")
    lines.append(f"- Recall: `{best_overall_row['recall']:.4f}`")
    lines.append(f"- Precision: `{best_overall_row['precision']:.4f}`")
    lines.append(f"- F1-score: `{best_overall_row['f1']:.4f}`")
    if best_comparator_row is not None:
        lines.append("")
        lines.append("## Best Comparator Model")
        lines.append("")
        lines.append(f"- Model: `{best_comparator_row['model']}`")
        lines.append(f"- Feature set: `{best_comparator_row['feature_set']}`")
        lines.append(f"- Label: `{best_comparator_row['feature_set_label']}`")
        lines.append(f"- PR-AUC: `{best_comparator_row['pr_auc']:.4f}`")
        lines.append(f"- ROC-AUC: `{best_comparator_row['roc_auc']:.4f}`")
        lines.append(f"- Recall: `{best_comparator_row['recall']:.4f}`")
        lines.append(f"- Precision: `{best_comparator_row['precision']:.4f}`")
        lines.append(f"- F1-score: `{best_comparator_row['f1']:.4f}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Pipeline mengikuti struktur fitur aktual: kategorikal, numeric binary-like, dan numeric non-binary diperlakukan berbeda.")
    lines.append("- Scaling hanya diterapkan pada model yang sensitif terhadap skala, yaitu Logistic Regression, SVM, dan KNN.")
    lines.append("- SMOTE belum dimasukkan ke baseline utama dan lebih cocok diposisikan sebagai eksperimen pembanding terpisah.")
    lines.append("- Missing indicator eksplisit sekarang diharapkan sudah tersedia di final dataset, sehingga imputasi median di pipeline tidak lagi menambahkan indicator otomatis.")
    lines.append("- `best_model.pkl` selalu menyimpan model primary terbaik tanpa medication, agar model utama tetap selaras dengan tujuan penelitian.")
    if best_comparator_row is not None:
        lines.append("- `best_comparator_model.pkl` menyimpan model pembanding yang boleh memakai medication untuk kebutuhan analisis leakage/perbandingan.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Results CSV: `{out_dir / 'baseline_results.csv'}`")
    lines.append(f"- Feature metadata: `{out_dir / 'feature_metadata.json'}`")
    lines.append(f"- Best model pickle: `{out_dir / 'best_model.pkl'}`")
    if best_comparator_row is not None:
        lines.append(f"- Best comparator model pickle: `{out_dir / 'best_comparator_model.pkl'}`")
    save_markdown(out_dir / "report.md", "\n".join(lines))

    return {
        "dataset_path": dataset_path,
        "out_dir": out_dir,
        "results_path": out_dir / "baseline_results.csv",
        "report_path": out_dir / "report.md",
        "best_model_path": out_dir / "best_model.pkl",
        "best_experiment": best_primary_row,
        "best_primary_experiment": best_primary_row,
        "best_overall_experiment": best_overall_row,
        "best_comparator_experiment": best_comparator_row,
    }


def run_leakage_check(
    dataset_path: Path = APP_FINAL_DATASET_PATH,
    out_dir: Path = TRAIN_OUTPUT_DIR,
    selected_models: list[str] | None = None,
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    df = load_csv(dataset_path)
    _, metadata = build_feature_sets(df)
    leakage_feature_sets = build_leakage_feature_sets(metadata)

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        stratify=df["DMIndicator"],
        random_state=random_state,
    )
    y_train_ref = df.loc[train_idx, "DMIndicator"].astype(int)
    neg_pos_ratio = float((y_train_ref == 0).sum() / (y_train_ref == 1).sum())

    model_builders = make_model_builders(neg_pos_ratio=neg_pos_ratio, random_state=random_state)
    default_models = ["Logistic Regression", "Gradient Boosting", "XGBoost", "LightGBM"]
    if selected_models is None:
        model_builders = {name: model_builders[name] for name in default_models}
    else:
        model_builders = {name: builder for name, builder in model_builders.items() if name in selected_models}

    results: list[dict[str, Any]] = []
    for feature_set_name, feature_cols in leakage_feature_sets.items():
        for model_name, model_builder in model_builders.items():
            result, _ = evaluate_experiment(
                df=df,
                feature_set_name=feature_set_name,
                feature_cols=feature_cols,
                model_name=model_name,
                model_builder=model_builder,
                train_idx=train_idx,
                test_idx=test_idx,
            )
            result["experiment_key"] = f"{model_name}__{feature_set_name}"
            results.append(result)

    results_df = pd.DataFrame(results).sort_values(["model", "pr_auc", "recall"], ascending=[True, False, False]).reset_index(drop=True)
    leakage_results_path = out_dir / "leakage_check_results.csv"
    results_df.to_csv(leakage_results_path, index=False)

    pivot_pr = results_df.pivot(index="feature_set", columns="model", values="pr_auc")
    pivot_recall = results_df.pivot(index="feature_set", columns="model", values="recall")
    pivot_roc = results_df.pivot(index="feature_set", columns="model", values="roc_auc")

    summary_rows: list[dict[str, Any]] = []
    for model_name in model_builders:
        model_slice = results_df[results_df["model"] == model_name].set_index("feature_set")
        clean_pr = float(model_slice.loc["clean_core", "pr_auc"])
        full_pr = float(model_slice.loc["full", "pr_auc"])
        med_pr = float(model_slice.loc["with_medication", "pr_auc"])
        diag_pr = float(model_slice.loc["with_diagnosis", "pr_auc"])
        summary_rows.append(
            {
                "model": model_name,
                "clean_core_pr_auc": round(clean_pr, 4),
                "with_diagnosis_pr_auc": round(diag_pr, 4),
                "with_medication_pr_auc": round(med_pr, 4),
                "full_pr_auc": round(full_pr, 4),
                "delta_full_vs_clean": round(full_pr - clean_pr, 4),
                "delta_diag_vs_clean": round(diag_pr - clean_pr, 4),
                "delta_med_vs_clean": round(med_pr - clean_pr, 4),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("delta_full_vs_clean", ascending=False)
    summary_path = out_dir / "leakage_check_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    lines: list[str] = []
    lines.append("# Leakage Check Report")
    lines.append("")
    lines.append(f"- Dataset: `{dataset_path}`")
    lines.append(f"- Test size: `{test_size}`")
    lines.append(f"- Random state: `{random_state}`")
    lines.append("")
    lines.append("## Tujuan")
    lines.append("")
    lines.append("- Mengecek seberapa besar kenaikan performa ketika fitur diagnosis dan medication mulai dimasukkan.")
    lines.append("- Membantu membedakan sinyal prediktif yang relatif bersih dari sinyal yang mungkin terlalu dekat dengan label diabetes.")
    lines.append("")
    lines.append("## Feature Set Leakage Check")
    lines.append("")
    lines.append("- `clean_core`: demografi + transcript")
    lines.append("- `clean_plus_physician`: clean_core + physician")
    lines.append("- `with_diagnosis`: clean_core + diagnosis")
    lines.append("- `with_medication`: clean_core + medication")
    lines.append("- `with_diagnosis_physician`: clean_core + diagnosis + physician")
    lines.append("- `with_physician_medication`: clean_core + physician + medication")
    lines.append("- `full`: clean_core + diagnosis + physician + medication")
    lines.append("")
    lines.append("## Summary per Model (PR-AUC deltas)")
    lines.append("")
    for row in summary_df.to_dict(orient="records"):
        lines.append(
            f"- `{row['model']}`: clean=`{row['clean_core_pr_auc']}`, diagnosis=`{row['with_diagnosis_pr_auc']}`, "
            f"medication=`{row['with_medication_pr_auc']}`, full=`{row['full_pr_auc']}`, "
            f"delta_full_clean=`{row['delta_full_vs_clean']}`, delta_diag_clean=`{row['delta_diag_vs_clean']}`, "
            f"delta_med_clean=`{row['delta_med_vs_clean']}`"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Results CSV: `{leakage_results_path}`")
    lines.append(f"- Summary CSV: `{summary_path}`")
    lines.append("")
    lines.append("## Catatan Interpretasi")
    lines.append("")
    lines.append("- Jika kenaikan performa dari `clean_core` ke `with_diagnosis` atau `with_medication` sangat besar, maka ada indikasi bahwa model memanfaatkan fitur yang sangat dekat dengan label.")
    lines.append("- Kenaikan kecil dan konsisten biasanya lebih mudah dipertanggungjawabkan daripada lonjakan sangat tajam.")
    save_markdown(out_dir / "leakage_check_report.md", "\n".join(lines))

    return {
        "dataset_path": dataset_path,
        "out_dir": out_dir,
        "results_path": leakage_results_path,
        "summary_path": summary_path,
        "report_path": out_dir / "leakage_check_report.md",
        "pr_auc_pivot": pivot_pr,
        "recall_pivot": pivot_recall,
        "roc_auc_pivot": pivot_roc,
        "summary_df": summary_df,
    }


def run_feature_leakage_audit(
    dataset_path: Path = APP_FINAL_DATASET_PATH,
    out_dir: Path = TRAIN_OUTPUT_DIR,
    min_support: int = 10,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    df = load_csv(dataset_path)
    _, metadata = build_feature_sets(df)

    diagnosis_stats = compute_presence_stats(df, metadata["diagnosis"], min_support=1)
    physician_stats = compute_presence_stats(df, metadata["physician"], min_support=1)
    medication_stats = compute_presence_stats(df, metadata["medication"], min_support=1)

    medication_keyword_hits = collect_keyword_hits(
        medication_stats,
        {
            "direct_diabetes_medication": DIRECT_DIABETES_MEDICATION_TERMS,
            "diabetes_supply_or_testing": DIABETES_SUPPLY_TERMS,
        },
    )
    physician_keyword_hits = collect_keyword_hits(
        physician_stats,
        {
            "endocrine_or_diabetes_specialty": ["endocrinology", "diabetes", "metabolism"],
        },
    )

    high_association_medication = medication_stats[medication_stats["support"] >= min_support].copy()
    high_association_medication = high_association_medication[
        high_association_medication["dm_rate_when_present"] >= 0.45
    ].reset_index(drop=True)

    high_risk_drop_candidates = medication_keyword_hits[
        medication_keyword_hits["matched_groups"].str.contains("direct_diabetes_medication|diabetes_supply_or_testing", regex=True)
    ].copy()
    high_risk_drop_candidates = high_risk_drop_candidates[high_risk_drop_candidates["support"] > 0].reset_index(drop=True)

    diagnosis_path = out_dir / "leakage_audit_diagnosis.csv"
    physician_path = out_dir / "leakage_audit_physician.csv"
    medication_path = out_dir / "leakage_audit_medication.csv"
    keyword_hits_path = out_dir / "leakage_audit_keyword_hits.csv"
    high_association_path = out_dir / "leakage_audit_high_association_medication.csv"

    diagnosis_stats.to_csv(diagnosis_path, index=False)
    physician_stats.to_csv(physician_path, index=False)
    medication_stats.to_csv(medication_path, index=False)
    medication_keyword_hits.to_csv(keyword_hits_path, index=False)
    high_association_medication.to_csv(high_association_path, index=False)

    base_dm_rate = float(df["DMIndicator"].mean())
    summary_payload = {
        "dataset_path": str(dataset_path),
        "base_dm_rate": base_dm_rate,
        "diagnosis_top_10": diagnosis_stats.head(10).to_dict(orient="records"),
        "physician_top_10": physician_stats.head(10).to_dict(orient="records"),
        "medication_top_20": medication_stats.head(20).to_dict(orient="records"),
        "medication_keyword_hits": medication_keyword_hits.to_dict(orient="records"),
        "physician_keyword_hits": physician_keyword_hits.to_dict(orient="records"),
        "high_association_medication": high_association_medication.head(30).to_dict(orient="records"),
        "high_risk_drop_candidates": high_risk_drop_candidates.to_dict(orient="records"),
    }
    save_json(out_dir / "leakage_audit_summary.json", summary_payload)

    lines: list[str] = []
    lines.append("# Feature Leakage Audit")
    lines.append("")
    lines.append("## Ringkasan")
    lines.append("")
    lines.append(f"- Dataset: `{dataset_path}`")
    lines.append(f"- Jumlah baris: `{len(df)}`")
    lines.append(f"- Prevalensi DM (`DMIndicator=1`): `{base_dm_rate:.4f}`")
    lines.append(f"- Diagnosis feature count: `{len(metadata['diagnosis'])}`")
    lines.append(f"- Physician feature count: `{len(metadata['physician'])}`")
    lines.append(f"- Medication feature count: `{len(metadata['medication'])}`")
    lines.append("")
    lines.append("## Temuan Utama")
    lines.append("")
    lines.append("- Grup diagnosis yang paling sensitif adalah `Icd9_240-279`, yaitu kelompok endocrine/metabolic. Ini wajar sangat terkait dengan diabetes, sehingga perlu dianggap fitur berisiko menempel ke label.")
    lines.append("- Di blok medication ada token yang tampak seperti hasil tokenisasi resep/supply, bukan nama obat bersih. Contoh paling mencolok: `pad`, `alcohol`, `isopropyl`, `swab`, `pump`.")
    lines.append("- Token supply/testing di atas sangat berisiko menjadi leakage karena lebih menggambarkan proses monitoring/penanganan diabetes daripada kondisi klinis umum pasien.")
    lines.append("- Banyak medication feature lain yang sangat terkait dengan DM, tetapi belum tentu leakage langsung. Sebagian besar bisa jadi menangkap komorbiditas kardiometabolik yang memang sering berjalan bersama diabetes.")
    lines.append("")
    lines.append("## Diagnosis Paling Terkait")
    lines.append("")
    for row in diagnosis_stats.head(10).to_dict(orient="records"):
        lines.append(
            f"- `{row['feature']}`: support=`{int(row['support'])}`, dm_rate_when_present=`{row['dm_rate_when_present']:.4f}`, lift_vs_base=`{row['lift_vs_base']:.2f}`"
        )
    lines.append("")
    lines.append("## Physician yang Perlu Diperhatikan")
    lines.append("")
    for row in physician_stats.head(10).to_dict(orient="records"):
        lines.append(
            f"- `{row['feature']}`: support=`{int(row['support'])}`, dm_rate_when_present=`{row['dm_rate_when_present']:.4f}`, lift_vs_base=`{row['lift_vs_base']:.2f}`"
        )
    lines.append("")
    lines.append("## Medication Keyword Hits")
    lines.append("")
    if medication_keyword_hits.empty:
        lines.append("- Tidak ada keyword hit yang terdeteksi.")
    else:
        for row in medication_keyword_hits.to_dict(orient="records"):
            lines.append(
                f"- `{row['feature']}` [{row['matched_groups']}]: support=`{int(row['support'])}`, dm_rate_when_present=`{row['dm_rate_when_present']:.4f}`, lift_vs_base=`{row['lift_vs_base']:.2f}`"
            )
    lines.append("")
    lines.append("## Medication dengan Asosiasi Tinggi")
    lines.append("")
    for row in high_association_medication.head(20).to_dict(orient="records"):
        lines.append(
            f"- `{row['feature']}`: support=`{int(row['support'])}`, dm_rate_when_present=`{row['dm_rate_when_present']:.4f}`, lift_vs_base=`{row['lift_vs_base']:.2f}`"
        )
    lines.append("")
    lines.append("## Rekomendasi Praktis")
    lines.append("")
    lines.append("- Drop lebih dulu medication token yang jelas merupakan diabetes treatment/testing/supply, terutama `pad`, `alcohol`, `isopropyl`, `swab`, `pump`, serta nama obat diabetes yang muncul langsung.")
    lines.append("- Jangan langsung membuang seluruh blok diagnosis, tetapi buat eksperimen pembanding tanpa `Icd9_240-279` agar laporan bisa menunjukkan sensitivitas model terhadap fitur endocrine/metabolic.")
    lines.append("- Pertahankan physician sebagai eksperimen pembanding, tetapi tidak perlu dijadikan pusat model karena kontribusinya kecil pada leakage check sebelumnya.")
    lines.append("- Sebelum masuk XAI, siapkan minimal dua model final: `cleaner_model` dan `full_model`, supaya interpretasi tidak terjebak menjelaskan fitur yang terlalu dekat dengan label.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Diagnosis stats: `{diagnosis_path}`")
    lines.append(f"- Physician stats: `{physician_path}`")
    lines.append(f"- Medication stats: `{medication_path}`")
    lines.append(f"- Medication keyword hits: `{keyword_hits_path}`")
    lines.append(f"- High association medication: `{high_association_path}`")
    save_markdown(out_dir / "leakage_audit_report.md", "\n".join(lines))

    return {
        "dataset_path": dataset_path,
        "out_dir": out_dir,
        "diagnosis_path": diagnosis_path,
        "physician_path": physician_path,
        "medication_path": medication_path,
        "keyword_hits_path": keyword_hits_path,
        "high_association_path": high_association_path,
        "report_path": out_dir / "leakage_audit_report.md",
        "summary_path": out_dir / "leakage_audit_summary.json",
    }


def run_smote_comparison(
    dataset_path: Path = APP_FINAL_DATASET_PATH,
    out_dir: Path = TRAIN_OUTPUT_DIR,
    selected_feature_sets: list[str] | None = None,
    selected_models: list[str] | None = None,
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    df = load_csv(dataset_path)
    feature_sets, _ = build_feature_sets(df)
    if selected_feature_sets:
        feature_sets = {name: cols for name, cols in feature_sets.items() if name in selected_feature_sets}

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        stratify=df["DMIndicator"],
        random_state=random_state,
    )
    y_train_ref = df.loc[train_idx, "DMIndicator"].astype(int)
    neg_pos_ratio = float((y_train_ref == 0).sum() / (y_train_ref == 1).sum())

    model_builders = make_model_builders(neg_pos_ratio=neg_pos_ratio, random_state=random_state)
    if selected_models is None:
        model_builders = {name: model_builders[name] for name in DEFAULT_SMOTE_MODELS}
    else:
        model_builders = {name: builder for name, builder in model_builders.items() if name in selected_models}

    results: list[dict[str, Any]] = []
    for feature_set_name, feature_cols in feature_sets.items():
        strategy = describe_feature_set(feature_set_name)
        for model_name, model_builder in model_builders.items():
            for use_smote in [False, True]:
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
                result["experiment_key"] = f"{model_name}__{feature_set_name}__{result['sampling_strategy']}"
                result["feature_set_label"] = strategy["label"]
                result["feature_set_role"] = strategy["role"]
                results.append(result)

    results_df = pd.DataFrame(results).sort_values(
        ["feature_set", "model", "sampling_strategy"], ascending=[True, True, True]
    ).reset_index(drop=True)
    results_path = out_dir / "smote_comparison_results.csv"
    results_df.to_csv(results_path, index=False)

    comparison_rows: list[dict[str, Any]] = []
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
    lines.append("## Tujuan")
    lines.append("")
    lines.append("- Membandingkan performa model tanpa SMOTE dan dengan SMOTE pada keempat feature set.")
    lines.append("- Fokus evaluasi pada perubahan `Recall`, `Precision`, `F1`, dan `PR-AUC`.")
    lines.append("")
    lines.append("## Feature Set yang Diuji")
    lines.append("")
    for feature_set_name in feature_sets:
        strategy = describe_feature_set(feature_set_name)
        lines.append(
            f"- `{feature_set_name}` ({strategy['label']} | role=`{strategy['role']}`): {strategy['description']}"
        )
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
    lines.append("")
    lines.append("## Catatan")
    lines.append("")
    lines.append("- Pada eksperimen ini, parameter model dibiarkan sama. SMOTE ditambahkan hanya pada data train setelah preprocessing.")
    lines.append("- Hasil `full` tetap dibaca hati-hati karena set ini masih mengandung medication dan berperan sebagai comparator leakage.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Results CSV: `{results_path}`")
    lines.append(f"- Summary CSV: `{comparison_path}`")
    report_path = out_dir / "smote_comparison_report.md"
    save_markdown(report_path, "\n".join(lines))

    return {
        "dataset_path": dataset_path,
        "out_dir": out_dir,
        "results_path": results_path,
        "summary_path": comparison_path,
        "report_path": report_path,
        "results_df": results_df,
        "comparison_df": comparison_df,
    }


def run_weighting_comparison(
    dataset_path: Path = APP_FINAL_DATASET_PATH,
    out_dir: Path = TRAIN_OUTPUT_DIR,
    selected_feature_sets: list[str] | None = None,
    selected_models: list[str] | None = None,
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    df = load_csv(dataset_path)
    feature_sets, _ = build_feature_sets(df)
    if selected_feature_sets:
        feature_sets = {name: cols for name, cols in feature_sets.items() if name in selected_feature_sets}

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

    if selected_models is None:
        model_names = DEFAULT_WEIGHTING_MODELS
    else:
        model_names = [name for name in DEFAULT_WEIGHTING_MODELS if name in selected_models]

    results: list[dict[str, Any]] = []
    for feature_set_name, feature_cols in feature_sets.items():
        strategy = describe_feature_set(feature_set_name)
        for model_name in model_names:
            for weighting_mode, model_builder, use_class_balancing in [
                ("weighted", weighted_builders[model_name], True),
                ("unweighted", unweighted_builders[model_name], False),
            ]:
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
                result["weighting_mode"] = weighting_mode
                result["experiment_key"] = f"{model_name}__{feature_set_name}__{weighting_mode}"
                result["feature_set_label"] = strategy["label"]
                result["feature_set_role"] = strategy["role"]
                results.append(result)

    results_df = pd.DataFrame(results).sort_values(
        ["feature_set", "model", "weighting_mode"], ascending=[True, True, True]
    ).reset_index(drop=True)
    results_path = out_dir / "weighting_comparison_results.csv"
    results_df.to_csv(results_path, index=False)

    comparison_rows: list[dict[str, Any]] = []
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
    lines.append("## Tujuan")
    lines.append("")
    lines.append("- Membandingkan performa model weighted vs unweighted pada feature set yang dipilih.")
    lines.append("- Weighted di sini berarti model menggunakan mekanisme penanganan imbalance bawaan model atau sample weight.")
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
    lines.append("")
    lines.append("## Catatan")
    lines.append("")
    lines.append("- `Logistic Regression` dan `SVM` weighted memakai `class_weight='balanced'`.")
    lines.append("- `XGBoost` weighted memakai `scale_pos_weight` berdasarkan rasio kelas di train.")
    lines.append("- `LightGBM` weighted memakai `class_weight='balanced'`.")
    lines.append("- `Gradient Boosting` weighted memakai `sample_weight` saat fitting.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Results CSV: `{results_path}`")
    lines.append(f"- Summary CSV: `{comparison_path}`")
    report_path = out_dir / "weighting_comparison_report.md"
    save_markdown(report_path, "\n".join(lines))

    return {
        "dataset_path": dataset_path,
        "out_dir": out_dir,
        "results_path": results_path,
        "summary_path": comparison_path,
        "report_path": report_path,
        "results_df": results_df,
        "comparison_df": comparison_df,
    }


def run_threshold_tuning(
    dataset_path: Path = APP_FINAL_DATASET_PATH,
    out_dir: Path = TRAIN_OUTPUT_DIR,
    selected_feature_sets: list[str] | None = None,
    selected_models: list[str] | None = None,
    thresholds: list[float] | None = None,
    test_size: float = 0.3,
    random_state: int = 42,
    use_class_balancing: bool = True,
) -> dict[str, Any]:
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
    model_builders = make_model_builders(
        neg_pos_ratio=neg_pos_ratio,
        random_state=random_state,
        use_class_balancing=use_class_balancing,
    )

    if selected_feature_sets is None and selected_models is None:
        experiments = list(DEFAULT_THRESHOLD_EXPERIMENTS)
    else:
        feature_set_names = selected_feature_sets or list(feature_sets)
        model_names = selected_models or DEFAULT_WEIGHTING_MODELS
        experiments = [(feature_set_name, model_name) for feature_set_name in feature_set_names for model_name in model_names]

    thresholds_to_use = sorted(set(thresholds or DEFAULT_THRESHOLDS))
    threshold_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []

    for feature_set_name, model_name in experiments:
        if feature_set_name not in feature_sets:
            raise ValueError(f"Feature set `{feature_set_name}` tidak ditemukan.")
        if model_name not in model_builders:
            raise ValueError(f"Model `{model_name}` tidak ditemukan.")

        strategy = describe_feature_set(feature_set_name)
        fit_result = fit_experiment_pipeline(
            df=df,
            feature_cols=feature_sets[feature_set_name],
            model_name=model_name,
            model_builder=model_builders[model_name],
            train_idx=train_idx,
            test_idx=test_idx,
            use_class_balancing=use_class_balancing,
        )

        experiment_rows: list[dict[str, Any]] = []
        for threshold in thresholds_to_use:
            metrics = compute_threshold_metrics(
                y_true=fit_result["y_test"],
                y_score=fit_result["y_score"],
                threshold=threshold,
            )
            metrics["feature_set"] = feature_set_name
            metrics["model"] = model_name
            metrics["feature_set_label"] = strategy["label"]
            metrics["feature_set_role"] = strategy["role"]
            metrics["use_class_balancing"] = use_class_balancing
            metrics["n_features"] = len(feature_sets[feature_set_name])
            metrics["experiment_key"] = f"{model_name}__{feature_set_name}__threshold_{threshold:.2f}"
            experiment_rows.append(metrics)

        experiment_df = pd.DataFrame(experiment_rows).sort_values(["f1", "recall", "pr_auc"], ascending=False).reset_index(drop=True)
        best_row = experiment_df.iloc[0].to_dict()
        best_rows.append(best_row)
        threshold_rows.extend(experiment_rows)

    threshold_df = pd.DataFrame(threshold_rows).sort_values(
        ["feature_set", "model", "threshold"], ascending=[True, True, True]
    ).reset_index(drop=True)
    threshold_results_path = out_dir / "threshold_tuning_results.csv"
    threshold_df.to_csv(threshold_results_path, index=False)

    best_df = pd.DataFrame(best_rows).sort_values(["f1", "recall", "pr_auc"], ascending=False).reset_index(drop=True)
    best_summary_path = out_dir / "threshold_tuning_best_thresholds.csv"
    best_df.to_csv(best_summary_path, index=False)

    lines: list[str] = []
    lines.append("# Threshold Tuning Report")
    lines.append("")
    lines.append(f"- Dataset: `{dataset_path}`")
    lines.append(f"- Test size: `{test_size}`")
    lines.append(f"- Random state: `{random_state}`")
    lines.append(f"- Weighted model: `{use_class_balancing}`")
    lines.append(f"- Threshold grid: `{', '.join(f'{value:.2f}' for value in thresholds_to_use)}`")
    lines.append("")
    lines.append("## Eksperimen")
    lines.append("")
    for feature_set_name, model_name in experiments:
        strategy = describe_feature_set(feature_set_name)
        lines.append(f"- `{feature_set_name}` + `{model_name}` ({strategy['label']} | role=`{strategy['role']}`)")
    lines.append("")
    lines.append("## Best Threshold per Experiment")
    lines.append("")
    for row in best_df.to_dict(orient="records"):
        lines.append(
            f"- `{row['feature_set']}` + `{row['model']}`: threshold=`{row['threshold']:.2f}`, "
            f"Recall=`{row['recall']:.4f}`, Precision=`{row['precision']:.4f}`, F1=`{row['f1']:.4f}`, PR-AUC=`{row['pr_auc']:.4f}`"
        )
    lines.append("")
    lines.append("## Catatan")
    lines.append("")
    lines.append("- PR-AUC dan ROC-AUC berasal dari skor probabilitas model, sehingga nilainya tetap untuk semua threshold dalam eksperimen yang sama.")
    lines.append("- Threshold tuning dipakai untuk mencari trade-off recall dan precision yang paling sesuai dengan tujuan deteksi diabetes.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Threshold results CSV: `{threshold_results_path}`")
    lines.append(f"- Best threshold summary CSV: `{best_summary_path}`")
    report_path = out_dir / "threshold_tuning_report.md"
    save_markdown(report_path, "\n".join(lines))

    return {
        "dataset_path": dataset_path,
        "out_dir": out_dir,
        "results_path": threshold_results_path,
        "summary_path": best_summary_path,
        "report_path": report_path,
        "threshold_df": threshold_df,
        "best_df": best_df,
    }
