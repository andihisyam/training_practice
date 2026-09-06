# Final Evaluation Report

- Created at: `2026-09-06T08:56:45+07:00`
- Dataset: `D:\Ratih\PracticeFusion\outputs\app\prepare_data\final_dataset.csv`
- Development/Test split: `70/30` dengan random_state `42`
- Random state: `42`
- Weighted model: `True`
- Optuna params dipakai bila tersedia: `True`

## Posisi Tahap Ini

- Tahap ini diposisikan sebagai evaluasi final pada `locked test` setelah keputusan model, feature set, imbalance handling, dan threshold sudah dibuat di development set.

## Struktur Feature Set

- `clinical_core_extreme` (Set B - Clinical Core + Extreme | role=`primary`): Kandidat utama: Age, Gender, BMI mean/max, systolic mean/max, dan diastolic mean/max.

## Best Primary Model

- Model: `XGBoost`
- Feature set: `clinical_core_extreme`
- Label: `Set B - Clinical Core + Extreme`
- PR-AUC: `0.4222`
- ROC-AUC: `0.7678`
- Recall: `0.8737`
- Specificity: `0.4915`
- Precision: `0.2894`
- F2-score: `0.6223`
- Threshold: `0.35`
- Brier score: `0.1927`

## Best Overall Experiment

- Model: `XGBoost`
- Feature set: `clinical_core_extreme`
- Label: `Set B - Clinical Core + Extreme`
- Role: `primary`
- PR-AUC: `0.4222`
- ROC-AUC: `0.7678`
- Recall: `0.8737`
- Specificity: `0.4915`
- Precision: `0.2894`
- F2-score: `0.6223`
- Threshold: `0.35`

## Notes

- Pipeline mengikuti struktur fitur aktual: kategorikal, numeric binary-like, dan numeric non-binary diperlakukan berbeda.
- Kolom konstan dibuang berdasarkan data development/train, bukan berdasarkan seluruh dataset.
- Scaling hanya diterapkan pada model yang sensitif terhadap skala, yaitu Logistic Regression, SVM, dan KNN.
- Threshold final diambil dari hasil threshold tuning jika tersedia; kalau belum ada, dipakai default `0.50`.
- Missing indicator eksplisit sekarang diharapkan sudah tersedia di final dataset, sehingga imputasi median di pipeline tidak lagi menambahkan indicator otomatis.
- `best_model.pkl` selalu menyimpan model primary terbaik tanpa medication, agar model utama tetap selaras dengan tujuan penelitian.

## Artifacts

- Split manifest: `D:\Ratih\PracticeFusion\outputs\app\train\locked_split_manifest.json`
- Results CSV: `D:\Ratih\PracticeFusion\outputs\app\train\final_test_results.csv`
- Final predictions CSV: `D:\Ratih\PracticeFusion\outputs\app\train\final_test_predictions.csv`
- Results CSV: `D:\Ratih\PracticeFusion\outputs\app\train\baseline_results.csv`
- Feature metadata: `D:\Ratih\PracticeFusion\outputs\app\train\feature_metadata.json`
- Best model pickle: `D:\Ratih\PracticeFusion\outputs\app\train\best_model.pkl`