# Threshold Tuning Report

- Created at: `2026-09-06T08:56:24+07:00`
- Dataset: `D:\Ratih\PracticeFusion\outputs\app\prepare_data\final_dataset.csv`
- Development/Test split: `70/30` dengan random_state `42`
- Development CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Weighted model: `True`
- Optuna params dipakai bila tersedia: `True`
- Threshold grid: `0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60`

## Eksperimen

- `clinical_core_extreme` + `XGBoost` (Set B - Clinical Core + Extreme | role=`primary`)

## Best Threshold per Experiment

- `clinical_core_extreme` + `XGBoost`: threshold=`0.35`, Recall=`0.8781`, Precision=`0.2919`, Specificity=`0.4955`, F2=`0.6265`, PR-AUC=`0.4370`

## Catatan

- Threshold dicari dari out-of-fold prediction pada development set, jadi `locked test` tidak ikut dipakai saat optimasi threshold.
- Urutan pemilihan threshold memprioritaskan `F2`, lalu `recall`, lalu `precision`, agar lebih selaras dengan konteks screening.

## Artifacts

- Split manifest: `D:\Ratih\PracticeFusion\outputs\app\train\locked_split_manifest.json`
- Threshold results CSV: `D:\Ratih\PracticeFusion\outputs\app\train\threshold_tuning_results.csv`
- OOF prediction CSV: `D:\Ratih\PracticeFusion\outputs\app\train\threshold_tuning_oof_predictions.csv`
- Best threshold summary CSV: `D:\Ratih\PracticeFusion\outputs\app\train\threshold_tuning_best_thresholds.csv`