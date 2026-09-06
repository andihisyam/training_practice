# XAI Report

- Created at: `2026-09-06T08:57:11+07:00`
- Dataset: `D:\Ratih\PracticeFusion\outputs\app\prepare_data\final_dataset.csv`
- Feature set: `clinical_core_extreme` (Set B - Clinical Core + Extreme)
- Model: `XGBoost`
- Threshold klasifikasi: `0.35`
- Development/Test split: `70/30` dengan random_state `42`
- Random state: `42`
- Weighted model: `True`
- Optuna params dipakai bila tersedia: `True`

## Ringkasan Sesuai Proposal

- SHAP digunakan pada model final untuk membaca kontribusi fitur secara global dan lokal.
- Output yang disiapkan mengikuti arah proposal: `summary plot`, `dependence plot`, dan `force plot` untuk pasien individual.
- Interpretasi SHAP dibaca sebagai kontribusi model, bukan bukti hubungan sebab-akibat.

## Fitur Paling Berpengaruh

- `Age`: mean |SHAP| = `0.8103`
- `BMI_Max`: mean |SHAP| = `0.2950`
- `DiastolicBP_Mean`: mean |SHAP| = `0.2509`
- `BMI_Mean`: mean |SHAP| = `0.2488`
- `SystolicBP_Max`: mean |SHAP| = `0.2281`
- `Gender_F`: mean |SHAP| = `0.1385`
- `SystolicBP_Mean`: mean |SHAP| = `0.0963`
- `DiastolicBP_Max`: mean |SHAP| = `0.0650`
- `Gender_M`: mean |SHAP| = `0.0347`

## Contoh Pasien Individual

- `tp_high_confidence` | error_type=`TP` | score=`0.9147` | top features: SystolicBP_Max, Age, DiastolicBP_Mean, BMI_Max, BMI_Mean
- `fn_borderline` | error_type=`FN` | score=`0.3483` | top features: BMI_Mean, Age, SystolicBP_Max, DiastolicBP_Mean, BMI_Max
- `fp_high_confidence` | error_type=`FP` | score=`0.9259` | top features: SystolicBP_Max, Age, BMI_Max, BMI_Mean, SystolicBP_Mean
- `tn_borderline` | error_type=`TN` | score=`0.3500` | top features: BMI_Max, BMI_Mean, Age, DiastolicBP_Mean, SystolicBP_Mean

## Artifacts

- Summary plot: `D:\Ratih\PracticeFusion\outputs\app\train\xai\shap_summary_plot.png`
- Summary bar plot: `D:\Ratih\PracticeFusion\outputs\app\train\xai\shap_summary_bar.png`
- Dependence plot: `D:\Ratih\PracticeFusion\outputs\app\train\xai\shap_dependence_Age.png`
- Dependence plot: `D:\Ratih\PracticeFusion\outputs\app\train\xai\shap_dependence_BMI_Max.png`
- Dependence plot: `D:\Ratih\PracticeFusion\outputs\app\train\xai\shap_dependence_DiastolicBP_Mean.png`
- Top feature CSV: `D:\Ratih\PracticeFusion\outputs\app\train\xai\xai_top_features.csv`
- Local case summary CSV: `D:\Ratih\PracticeFusion\outputs\app\train\xai\xai_local_cases_summary.csv`
- Local case `tp_high_confidence` CSV: `D:\Ratih\PracticeFusion\outputs\app\train\xai\xai_local_tp_high_confidence.csv`
- Local case `tp_high_confidence` force plot: `D:\Ratih\PracticeFusion\outputs\app\train\xai\xai_force_tp_high_confidence.html`
- Local case `fn_borderline` CSV: `D:\Ratih\PracticeFusion\outputs\app\train\xai\xai_local_fn_borderline.csv`
- Local case `fn_borderline` force plot: `D:\Ratih\PracticeFusion\outputs\app\train\xai\xai_force_fn_borderline.html`
- Local case `fp_high_confidence` CSV: `D:\Ratih\PracticeFusion\outputs\app\train\xai\xai_local_fp_high_confidence.csv`
- Local case `fp_high_confidence` force plot: `D:\Ratih\PracticeFusion\outputs\app\train\xai\xai_force_fp_high_confidence.html`
- Local case `tn_borderline` CSV: `D:\Ratih\PracticeFusion\outputs\app\train\xai\xai_local_tn_borderline.csv`
- Local case `tn_borderline` force plot: `D:\Ratih\PracticeFusion\outputs\app\train\xai\xai_force_tn_borderline.html`