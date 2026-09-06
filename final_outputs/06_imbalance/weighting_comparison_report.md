# Weighting Comparison Report

- Created at: `2026-09-06T00:55:42+07:00`
- Dataset: `D:\Ratih\PracticeFusion\outputs\app\prepare_data\final_dataset.csv`
- Development/Test split: `70/30` dengan random_state `42`
- CV di development: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`

## Tujuan

- Membandingkan performa model weighted vs unweighted pada development CV.
- Weighted di sini berarti model menggunakan mekanisme penanganan imbalance bawaan model atau sample weight.
- Perbandingan ini dijalankan tanpa SMOTE agar efek class weighting bisa dibaca secara terpisah.

## Ringkasan Perbandingan

- `clinical_core_extreme` + `XGBoost`: PR-AUC weighted `0.4393` vs unweighted `0.4366` (delta `0.0028`), Recall weighted `0.7194` vs unweighted `0.1475` (delta `0.5719`), Precision weighted `0.3409` vs unweighted `0.5499` (delta `-0.209`), F2 weighted `0.5886` vs unweighted `0.1727` (delta `0.4158`)

## Catatan

- `Logistic Regression` dan `SVM` weighted memakai `class_weight='balanced'`.
- `XGBoost` weighted memakai `scale_pos_weight` berdasarkan rasio kelas di train.
- `LightGBM` weighted memakai `class_weight='balanced'`.
- `Gradient Boosting` weighted memakai `sample_weight` saat fitting.

## Artifacts

- Split manifest: `D:\Ratih\PracticeFusion\outputs\app\train\locked_split_manifest.json`
- Results CSV: `D:\Ratih\PracticeFusion\outputs\app\train\weighting_comparison_results.csv`
- Summary CSV: `D:\Ratih\PracticeFusion\outputs\app\train\weighting_comparison_summary.csv`