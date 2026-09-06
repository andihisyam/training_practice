# Feature Set Screening Report

- Created at: `2026-09-06T10:34:55+07:00`
- Dataset: `D:\Ratih\PracticeFusion\outputs\app\prepare_data\final_dataset.csv`
- Development/Test split: `70/30` dengan random_state `42`
- Development CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Model anchor: `Logistic Regression, Random Forest`
- Weighted model: `True`

## Tujuan

- Membandingkan beberapa feature set pada development CV agar pemilihan fitur tidak diputuskan dari locked test.
- Default screening memakai dua anchor, yaitu model linear dan model nonlinear, supaya keputusan feature set tidak bergantung pada satu jenis algoritma saja.
- `Random Forest` di tahap ini hanya dipakai sebagai anchor sensitivitas pemilihan fitur, bukan sebagai model utama pada perbandingan 6 algoritma.
- `Locked test` belum dipakai di tahap ini; semua keputusan masih dibuat di development set.

## Ringkasan Gabungan Anchor

- `full_transcript_comparator`: PR-AUC `0.4225 +/- 0.0248`, Recall `0.4417 +/- 0.0197`, F2 `0.3924 +/- 0.0176`, Brier `0.1651 +/- 0.0039`, anchor `Logistic Regression, Random Forest`, fitur efektif `37-37`
- `clinical_core_extreme_weight`: PR-AUC `0.4201 +/- 0.0269`, Recall `0.4718 +/- 0.0162`, F2 `0.4244 +/- 0.0151`, Brier `0.1667 +/- 0.0045`, anchor `Logistic Regression, Random Forest`, fitur efektif `10-10`
- `clinical_core_extreme`: PR-AUC `0.4164 +/- 0.0301`, Recall `0.4884 +/- 0.0187`, F2 `0.4407 +/- 0.0184`, Brier `0.1673 +/- 0.0053`, anchor `Logistic Regression, Random Forest`, fitur efektif `8-8`
- `clinical_core`: PR-AUC `0.3948 +/- 0.0338`, Recall `0.4917 +/- 0.0292`, F2 `0.4403 +/- 0.0275`, Brier `0.1711 +/- 0.0056`, anchor `Logistic Regression, Random Forest`, fitur efektif `5-5`

## Ringkasan per Anchor

- `full_transcript_comparator` dengan `Random Forest`: PR-AUC `0.4252 +/- 0.0140`, Recall `0.1618 +/- 0.0140`, F2 `0.1876 +/- 0.0151`, Brier `0.1337 +/- 0.0014`, fitur efektif `37-37`
- `clinical_core_extreme_weight` dengan `Random Forest`: PR-AUC `0.4238 +/- 0.0171`, Recall `0.2333 +/- 0.0151`, F2 `0.2610 +/- 0.0153`, Brier `0.1361 +/- 0.0030`, fitur efektif `10-10`
- `full_transcript_comparator` dengan `Logistic Regression`: PR-AUC `0.4198 +/- 0.0357`, Recall `0.7216 +/- 0.0253`, F2 `0.5973 +/- 0.0201`, Brier `0.1966 +/- 0.0064`, fitur efektif `37-37`
- `clinical_core_extreme` dengan `Logistic Regression`: PR-AUC `0.4175 +/- 0.0349`, Recall `0.7103 +/- 0.0147`, F2 `0.5882 +/- 0.0132`, Brier `0.1971 +/- 0.0060`, fitur efektif `8-8`
- `clinical_core_extreme_weight` dengan `Logistic Regression`: PR-AUC `0.4164 +/- 0.0367`, Recall `0.7103 +/- 0.0174`, F2 `0.5878 +/- 0.0149`, Brier `0.1974 +/- 0.0061`, fitur efektif `10-10`
- `clinical_core_extreme` dengan `Random Forest`: PR-AUC `0.4153 +/- 0.0252`, Recall `0.2664 +/- 0.0226`, F2 `0.2932 +/- 0.0236`, Brier `0.1376 +/- 0.0045`, fitur efektif `8-8`
- `clinical_core` dengan `Logistic Regression`: PR-AUC `0.4045 +/- 0.0344`, Recall `0.7118 +/- 0.0263`, F2 `0.5873 +/- 0.0235`, Brier `0.1997 +/- 0.0060`, fitur efektif `5-5`
- `clinical_core` dengan `Random Forest`: PR-AUC `0.3852 +/- 0.0331`, Recall `0.2717 +/- 0.0321`, F2 `0.2933 +/- 0.0316`, Brier `0.1425 +/- 0.0053`, fitur efektif `5-5`

## Feature Set Teratas

- Feature set yang dipilih oleh aturan seleksi adalah `clinical_core_extreme` dengan PR-AUC `0.4164` dan recall `0.4884`.
- Alasan: Feature set ini dipilih karena performanya masih kompetitif terhadap PR-AUC tertinggi (gap 0.0061) dan lebih sederhana secara metodologis.

## Artifacts

- Split manifest: `D:\Ratih\PracticeFusion\outputs\app\train\locked_split_manifest.json`
- Fold-level results CSV: `D:\Ratih\PracticeFusion\outputs\app\train\feature_set_screening_fold_results.csv`
- Summary per anchor CSV: `D:\Ratih\PracticeFusion\outputs\app\train\feature_set_screening_summary.csv`
- Summary gabungan anchor CSV: `D:\Ratih\PracticeFusion\outputs\app\train\feature_set_screening_anchor_summary.csv`
- Selected feature set JSON: `D:\Ratih\PracticeFusion\outputs\app\train\selected_feature_set.json`