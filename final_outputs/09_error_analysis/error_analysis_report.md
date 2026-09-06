# Error Analysis Report

- Created at: `2026-09-06T08:56:56+07:00`
- Dataset: `D:\Ratih\PracticeFusion\outputs\app\prepare_data\final_dataset.csv`
- Feature set: `clinical_core_extreme` (Set B - Clinical Core + Extreme)
- Model: `XGBoost`
- Development/Test split: `70/30` dengan random_state `42`
- Random state: `42`
- Weighted model: `True`
- Optuna params dipakai bila tersedia: `True`
- Thresholds: `0.35, 0.50, 0.60`

## Ringkasan Threshold

- Threshold `0.35`: TP=`498`, TN=`1182`, FP=`1223`, FN=`72`, Recall=`0.8737`, Precision=`0.2894`, F1=`0.4347`
- Threshold `0.50`: TP=`412`, TN=`1607`, FP=`798`, FN=`158`, Recall=`0.7228`, Precision=`0.3405`, F1=`0.4629`
- Threshold `0.60`: TP=`317`, TN=`1891`, FP=`514`, FN=`253`, Recall=`0.5561`, Precision=`0.3815`, F1=`0.4525`

## Pola Error yang Perlu Diperhatikan

- Saat threshold dinaikkan dari `0.35` ke `0.60`, FP turun dari `1223` menjadi `514`, tetapi FN naik dari `72` menjadi `253`.
- Threshold `0.35` | `pasien DM yang terlewat`: `Age` `47.18` vs `65.54` (delta `-18.36`); `BMI_Mean` `28.54` vs `31.86` (delta `-3.31`)
- Threshold `0.35` | `pasien non-DM yang salah terdeteksi`: `Age` `64.87` vs `39.68` (delta `+25.18`); `BMI_Mean` `29.96` vs `26.28` (delta `+3.68`)
- Threshold `0.50` | `pasien DM yang terlewat`: `Age` `52.00` vs `67.52` (delta `-15.52`); `BMI_Mean` `29.18` vs `32.30` (delta `-3.12`)
- Threshold `0.50` | `pasien non-DM yang salah terdeteksi`: `Age` `67.36` vs `45.10` (delta `+22.26`); `BMI_Mean` `31.11` vs `26.69` (delta `+4.42`)
- Threshold `0.60` | `pasien DM yang terlewat`: `Age` `56.58` vs `68.52` (delta `-11.94`); `BMI_Mean` `29.39` vs `33.07` (delta `-3.69`)
- Threshold `0.60` | `pasien non-DM yang salah terdeteksi`: `Age` `68.65` vs `48.10` (delta `+20.55`); `BMI_Mean` `32.04` vs `27.10` (delta `+4.94`)

## Artifacts

- Summary CSV: `D:\Ratih\PracticeFusion\outputs\app\train\error_analysis_summary.csv`
- Predictions CSV: `D:\Ratih\PracticeFusion\outputs\app\train\error_analysis_predictions.csv`
- Numeric profile CSV: `D:\Ratih\PracticeFusion\outputs\app\train\error_analysis_numeric_profile.csv`
- Binary pattern CSV: `D:\Ratih\PracticeFusion\outputs\app\train\error_analysis_binary_patterns.csv`