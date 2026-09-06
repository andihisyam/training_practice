# SMOTE Comparison Report

- Created at: `2026-09-06T00:55:53+07:00`
- Dataset: `D:\Ratih\PracticeFusion\outputs\app\prepare_data\final_dataset.csv`
- Development/Test split: `70/30` dengan random_state `42`
- CV di development: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`

## Tujuan

- Membandingkan performa model tanpa SMOTE dan dengan SMOTE pada development CV.
- Fokus evaluasi pada perubahan `Recall`, `Precision`, `F1`, dan `PR-AUC`.
- Untuk menjaga eksperimen tetap fair, perbandingan ini memakai `weight = OFF` pada kedua sisi, sehingga efek yang dibaca benar-benar efek SMOTE.

## Feature Set yang Diuji

- `clinical_core_extreme` (Set B - Clinical Core + Extreme | role=`primary`): Kandidat utama: Age, Gender, BMI mean/max, systolic mean/max, dan diastolic mean/max.

## Ringkasan Perbandingan

- `clinical_core_extreme` + `XGBoost`: PR-AUC `0.4366` -> `0.4231` (delta `-0.0135`), Recall `0.1475` -> `0.5711` (delta `0.4236`), Precision `0.5499` -> `0.3873` (delta `-0.1625`), F2 `0.1727` -> `0.521` (delta `0.3483`)

## Catatan

- Pada eksperimen ini, parameter model dibiarkan sama. SMOTE ditambahkan hanya pada train fold setelah preprocessing.
- Hasil `full` tetap dibaca hati-hati karena set ini masih mengandung medication dan berperan sebagai comparator leakage.

## Artifacts

- Split manifest: `D:\Ratih\PracticeFusion\outputs\app\train\locked_split_manifest.json`
- Results CSV: `D:\Ratih\PracticeFusion\outputs\app\train\smote_comparison_results.csv`
- Summary CSV: `D:\Ratih\PracticeFusion\outputs\app\train\smote_comparison_summary.csv`