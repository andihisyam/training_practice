# Rebuild Research Summary

## Data

- Final dataset: `9916 x 2510`
- Label distribution:
  - `DMIndicator=0`: `8017` (`80.85%`)
  - `DMIndicator=1`: `1899` (`19.15%`)
- Cohort final sekarang memakai inner join `backbone d5 + transcript rebuilt`, lalu `diagnosis`, `physician`, dan `medication` digabung sebagai blok opsional dengan left join.

## Feature Set Strategy

- `Set A - Demografi + Transcript`: model utama paling bersih.
- `Set B - Demografi + Transcript + Physician`: model utama bersih dengan konteks layanan.
- `Set C - Demografi + Transcript + Diagnosis`: set sensitivitas untuk mengukur ketergantungan terhadap ICD.
- `Set D - Demografi + Transcript + Diagnosis + Physician`: set sensitivitas yang lebih dekat ke proses diagnosis.
- `Set E - Full + Medication`: comparator leakage.

## Baseline Training

- Best primary model: `Set B + XGBoost`
  - PR-AUC: `0.4381`
  - ROC-AUC: `0.7833`
  - Recall: `0.7298`
  - Precision: `0.3688`
  - F1-score: `0.4900`
- Best overall comparator: `Set E + LightGBM`
  - PR-AUC: `0.4925`
  - ROC-AUC: `0.8165`
  - Recall: `0.6912`
  - Precision: `0.4196`
  - F1-score: `0.5222`

## Leakage Check

- Penambahan diagnosis menaikkan PR-AUC cukup konsisten dari `clean_core`.
- Penambahan medication menaikkan performa lebih jauh pada model pohon, sehingga tetap diposisikan sebagai comparator leakage.
- Ringkasan `delta_full_vs_clean`:
  - `LightGBM`: `+0.0550`
  - `Gradient Boosting`: `+0.0448`
  - `XGBoost`: `+0.0387`
  - `Logistic Regression`: `-0.0249`

## Weighted vs Unweighted

- Fokus model utama `Set B + XGBoost`:
  - Weighted PR-AUC: `0.4381`
  - Unweighted PR-AUC: `0.4357`
  - Weighted Recall: `0.7298`
  - Unweighted Recall: `0.1561`
  - Weighted F1: `0.4900`
  - Unweighted F1: `0.2465`
- Keputusan: tetap pakai `weighted`.

## SMOTE vs No SMOTE

- Fokus model utama `Set B + XGBoost`:
  - No SMOTE PR-AUC: `0.4381`
  - SMOTE PR-AUC: `0.3950`
  - No SMOTE Recall: `0.7298`
  - SMOTE Recall: `0.8702`
  - No SMOTE F1: `0.4900`
  - SMOTE F1: `0.4509`
- Keputusan: `SMOTE` tidak dipakai sebagai default.

## Cross-Validation

- Best CV model: `Set B + LightGBM`
  - Mean PR-AUC: `0.4684 +/- 0.0167`
  - Mean Recall: `0.6741 +/- 0.0209`
  - Mean Precision: `0.3886 +/- 0.0109`
  - Mean F1: `0.4930 +/- 0.0140`
- `Set B + XGBoost` tetap sangat kompetitif:
  - Mean PR-AUC: `0.4652 +/- 0.0159`
  - Mean Recall: `0.7388 +/- 0.0218`
  - Mean Precision: `0.3629 +/- 0.0090`
  - Mean F1: `0.4867 +/- 0.0126`

## Threshold Tuning

- `Set B + XGBoost`: threshold terbaik tetap `0.50`
- `Set A + LightGBM`: threshold terbaik `0.45`
- Untuk konteks screening medis, threshold `0.50` tetap dipilih untuk model utama karena lebih seimbang terhadap risiko false negative.

## Error Analysis

- `Set B + XGBoost` pada threshold `0.50`:
  - TP: `416`
  - TN: `1693`
  - FP: `712`
  - FN: `154`
- Naik ke threshold `0.60`:
  - FP turun `712 -> 441`
  - FN naik `154 -> 264`
- Pola error utama:
  - `FN` cenderung lebih muda, berat badan lebih rendah, BMI lebih rendah.
  - `FP` cenderung lebih tua, berat badan lebih tinggi, BMI lebih tinggi.

## XAI

- Model XAI final: `Set B + XGBoost`, threshold `0.50`
- Top global SHAP features:
  1. `Age`
  2. `BMI_Mean`
  3. `BMI_Max`
  4. `DiastolicBP_Mean`
  5. `PhySp_Internal_Medicine`
- File gambar utama:
  - `shap_summary_plot.png`
  - `shap_summary_bar.png`
  - `shap_dependence_Age.png`
  - `shap_dependence_BMI_Mean.png`
  - `shap_dependence_BMI_Max.png`

## Final Position for Writing

- Model utama penelitian:
  - `Set B - Demografi + Transcript + Physician`
  - `XGBoost` untuk holdout utama
  - `LightGBM` sebagai pembanding CV yang sangat kuat
- ICD tidak dipakai pada model utama karena berisiko membuat model terlalu bergantung pada informasi diagnosis yang belum tentu tersedia pada pasien baru.
- Medication tetap diposisikan sebagai comparator leakage, bukan model utama.
