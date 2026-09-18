# Final Outputs Penelitian

Folder ini berisi artefak final yang layak dipush ke GitHub untuk dokumentasi hasil penelitian.

Isi folder ini sengaja dipilih agar GitHub tetap bersih:
- menyimpan report, tabel ringkasan, dan visualisasi penting
- tidak menyimpan dataset final besar
- tidak menyimpan daftar PatientGuid train/test
- tidak menyimpan file model `.pkl`
- tidak menyimpan raw prediction panjang
- tidak menyimpan plot lama yang masih memuat fitur ICD

## Struktur

### `01_prepare_data`

Berisi report proses pembentukan final dataset.

File penting:
- `prepare_report.md`
- `prepare_report.json`

### `02_eda`

Berisi ringkasan EDA dan plot eksplorasi data.

File penting:
- `report.md`
- `eda_summary.json`
- `plots/label_distribution.png`
- `plots/selected_boxplots.png`
- `plots/selected_correlation.png`
- `plots/selected_numeric_distribution.png`

### `03_methodology`

Berisi sanity check metodologi.

File penting:
- `methodology_checks_report.md`
- `methodology_checks.json`

### `04_feature_set_screening`

Berisi hasil pemilihan feature set pada development CV.

Default terbaru memakai dua anchor:
- `Logistic Regression`
- `Random Forest`

Random Forest hanya dipakai sebagai sensitivity check untuk pemilihan feature set, bukan sebagai model utama.

File penting:
- `feature_set_screening_report.md`
- `feature_set_screening_anchor_summary.csv`
- `feature_set_screening_summary.csv`
- `selected_feature_set.json`

### `05_model_comparison`

Berisi perbandingan 6 algoritma utama.

File penting:
- `main_cv_report.md`
- `main_cv_summary.csv`

### `06_imbalance`

Berisi eksperimen imbalance handling.

File penting:
- `weighting_comparison_report.md`
- `weighting_comparison_summary.csv`
- `weighting_comparison_results.csv`
- `smote_comparison_report.md`
- `smote_comparison_summary.csv`
- `smote_comparison_results.csv`

### `07_optuna_threshold`

Berisi hasil tuning hyperparameter dan threshold.

File penting:
- `optuna_report.md`
- `optuna_baseline_vs_tuned.csv`
- `optuna_best_params.json`
- `threshold_tuning_report.md`
- `threshold_tuning_best_thresholds.csv`
- `threshold_tuning_results.csv`

### `08_final_evaluation`

Berisi evaluasi final pada locked test.

File penting:
- `report.md`
- `final_test_results.csv`
- `final_test_predictions_for_curves.csv`
- `calibration_clinical_core_extreme_XGBoost.csv`
- `calibration_clinical_core_extreme_XGBoost.png`

`final_test_predictions_for_curves.csv` berisi `y_true` dan `y_score` tanpa `PatientGuid`, sehingga bisa dipakai untuk membuat ROC curve, PR curve, ROC-AUC, dan PR-AUC tanpa perlu mem-push dataset final besar.

### `09_error_analysis`

Berisi analisis error model final.

File penting:
- `error_analysis_report.md`
- `error_analysis_summary.csv`
- `error_analysis_numeric_profile.csv`
- `error_analysis_binary_patterns.csv`

### `10_xai`

Berisi interpretasi model dengan SHAP.

File penting:
- `xai_report.md`
- `xai_top_features.csv`
- `xai_local_cases_summary.csv`
- `plots/shap_summary_plot.png`
- `plots/shap_summary_bar.png`
- `plots/shap_dependence_Age.png`
- `plots/shap_dependence_BMI_Max.png`
- `plots/shap_dependence_BMI_Mean.png`
- `plots/shap_dependence_DiastolicBP_Mean.png`
- `force_plots/xai_force_*.html`

### `99_summary`

Berisi kesimpulan naratif hasil penelitian.

File penting:
- `general_conclusion.md`
- `rebuild_research_summary.md`

## Catatan

Kalau ingin menghasilkan ulang isi folder ini, jalankan pipeline dari `main.py`, lalu salin artefak final dari `outputs/app/...` ke folder ini.
