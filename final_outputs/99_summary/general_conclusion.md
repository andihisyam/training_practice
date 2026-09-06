# Kesimpulan Umum

## Gambaran Singkat

- Dataset final yang dipakai saat ini: `9916 x 2510`
- Model terbaik pada evaluasi holdout utama: `XGBoost`
- Feature set utama yang dipakai: `Set B - Demografi + Transcript + Physician`
- PR-AUC: `0.4381`
- Recall: `0.7298`
- Precision: `0.3688`
- F1-score: `0.4900`

## Data yang Dipakai

- Model final memakai `Set B - Demografi + Transcript + Physician`.
- Deskripsi singkat: Model utama tanpa ICD, dengan tambahan specialty dokter sebagai konteks layanan.
- Alasan memilih set ini: set ini dipilih karena tetap bersih dari ICD dan medication, tetapi masih memberi konteks layanan melalui specialty dokter.
- Diagnosis ICD tidak dipakai pada model utama agar model tidak menunggu informasi diagnosis yang belum tentu tersedia pada pasien baru.
- Medication tetap diposisikan sebagai pembanding leakage, bukan fondasi model utama.

## Model yang Paling Bagus

- Secara holdout, model utama terbaik adalah `XGBoost` pada `Set B - Demografi + Transcript + Physician` dengan PR-AUC `0.4381` dan recall `0.7298`.
- Secara cross-validation, model paling stabil saat ini adalah `LightGBM` pada `Set B - Demografi + Transcript + Physician` dengan mean PR-AUC `0.4684` dan mean F1 `0.4930`.
- Threshold terbaik untuk model utama saat ini adalah `0.50` dengan recall `0.7298` dan precision `0.3688`.

## Penjelasan XAI

- Fitur SHAP paling dominan pada model final adalah: Age, BMI_Mean, BMI_Max, DiastolicBP_Mean, PhySp_Internal_Medicine.
- Secara umum, model membaca usia, BMI, tekanan darah, dan konteks layanan sebagai sinyal utama untuk memprediksi diabetes.
- Ini membuat interpretasi model lebih klinis dan lebih mudah dipertahankan dibanding model yang terlalu bergantung pada ICD atau medication.

## Ringkasan Keputusan

- Model utama untuk penulisan: `XGBoost`
- Feature set utama untuk penulisan: `Set B - Demografi + Transcript + Physician`
- Threshold final: `0.50`
- Strategi imbalance: `weighted`
- SMOTE: tidak dipakai sebagai default
- XAI: menggunakan SHAP pada model final