# Main Cross-Validation Report

- Created at: `2026-09-06T00:55:25+07:00`
- Dataset: `D:\Ratih\PracticeFusion\outputs\app\prepare_data\final_dataset.csv`
- Development/Test split: `70/30` dengan random_state `42`
- CV strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` pada `development` saja
- Weighted model: `True`

## Tujuan

- Memberikan evaluasi yang lebih stabil untuk perbandingan algoritma dengan feature set yang sama.
- `Locked test` belum disentuh pada tahap ini, sehingga hasil CV dipakai untuk pemilihan model, bukan skor final.

## Feature Set yang Diuji

- `clinical_core_extreme` (Set B - Clinical Core + Extreme | role=`primary`): Kandidat utama: Age, Gender, BMI mean/max, systolic mean/max, dan diastolic mean/max.

## Ringkasan Hasil

- `clinical_core_extreme` + `XGBoost`: PR-AUC `0.4393 +/- 0.0312`, Recall `0.7194 +/- 0.0273`, Precision `0.3409 +/- 0.0147`, F2 `0.5886 +/- 0.0223`, ROC-AUC `0.7740 +/- 0.0150`, fitur efektif `8-8`
- `clinical_core_extreme` + `Gradient Boosting`: PR-AUC `0.4269 +/- 0.0238`, Recall `0.7276 +/- 0.0283`, Precision `0.3384 +/- 0.0106`, F2 `0.5915 +/- 0.0196`, ROC-AUC `0.7719 +/- 0.0112`, fitur efektif `8-8`
- `clinical_core_extreme` + `LightGBM`: PR-AUC `0.4216 +/- 0.0298`, Recall `0.6576 +/- 0.0124`, Precision `0.3526 +/- 0.0132`, F2 `0.5605 +/- 0.0111`, ROC-AUC `0.7648 +/- 0.0157`, fitur efektif `8-8`
- `clinical_core_extreme` + `Logistic Regression`: PR-AUC `0.4175 +/- 0.0349`, Recall `0.7103 +/- 0.0147`, Precision `0.3486 +/- 0.0100`, F2 `0.5882 +/- 0.0132`, ROC-AUC `0.7742 +/- 0.0150`, fitur efektif `8-8`
- `clinical_core_extreme` + `SVM`: PR-AUC `0.4153 +/- 0.0359`, Recall `0.7427 +/- 0.0198`, Precision `0.3442 +/- 0.0122`, F2 `0.6031 +/- 0.0176`, ROC-AUC `0.7738 +/- 0.0156`, fitur efektif `8-8`
- `clinical_core_extreme` + `KNN`: PR-AUC `0.3756 +/- 0.0266`, Recall `0.1798 +/- 0.0099`, Precision `0.4576 +/- 0.0554`, F2 `0.2045 +/- 0.0111`, ROC-AUC `0.7332 +/- 0.0175`, fitur efektif `8-8`

## Model CV Terbaik

- Feature set: `clinical_core_extreme`
- Model: `XGBoost`
- Mean PR-AUC: `0.4393`
- Mean Recall: `0.7194`
- Mean Precision: `0.3409`
- Mean F2: `0.5886`

## Catatan

- Nilai mean/std membantu melihat apakah performa model stabil di beberapa fold, bukan hanya kebetulan satu split.
- Tahap ini dimaksudkan untuk membandingkan algoritma pada ruang fitur yang sama, sehingga feature set default hanya satu.
- Feature set lain bisa diuji lewat `feature-set screening`, bukan dicampur dalam ranking utama algoritma.

## Artifacts

- Split manifest: `D:\Ratih\PracticeFusion\outputs\app\train\locked_split_manifest.json`
- Fold-level results CSV: `D:\Ratih\PracticeFusion\outputs\app\train\main_cv_fold_results.csv`
- Summary CSV: `D:\Ratih\PracticeFusion\outputs\app\train\main_cv_summary.csv`