# Panduan Sederhana `python main.py`

Dokumen ini menjelaskan cara menjalankan pipeline penelitian dari terminal atau menu interaktif.

Penelitian ini difokuskan pada **klasifikasi status Diabetes Mellitus Tipe 2 (T2DM)** berdasarkan data EHR yang sudah diringkas per pasien. Ini bukan prediksi kejadian diabetes di masa depan.

## Cara Paling Mudah

Jalankan:

```powershell
python main.py
```

Kalau dijalankan tanpa argumen, program menampilkan menu:

1. `Jalankan Pipeline / Eksperimen`
2. `Lihat Ringkasan Hasil`
3. `Buka Visualisasi / Report`
4. `Kesimpulan Umum`
0. `Keluar`

Menu ini dibuat agar dosen atau penguji bisa menjalankan dan melihat hasil tanpa harus membuka banyak folder.

## Command Utama

### 1. Prepare Data

```powershell
python main.py prepare-data
```

Membangun dataset final dari data mentah.

Output utama:
- `outputs/app/prepare_data/final_dataset.csv`
- `outputs/app/prepare_data/transcript_rebuilt.csv`
- `outputs/app/prepare_data/prepare_report.md`

### 2. EDA

```powershell
python main.py eda
```

Membuat ringkasan eksplorasi data.

Output utama:
- `outputs/app/eda/report.md`
- `outputs/app/eda/eda_summary.json`

### 3. Methodology Check

```powershell
python main.py methodology-check
```

Memeriksa apakah split dan feature set utama sudah aman secara metodologi.

Yang dicek:
- satu pasien hanya muncul satu baris
- development dan test tidak bercampur
- split reproducible dengan seed yang sama
- primary feature set tidak memakai `State`, `PhySp_*`, ICD, medication, `PatientGuid`, atau `PracticeGuid`

Output utama:
- `outputs/app/train/methodology_checks_report.md`
- `outputs/app/train/methodology_checks.json`

### 4. Feature Set Screening

```powershell
python main.py feature-set-screen
```

Membandingkan kandidat feature set dengan model anchor yang sama. Tahap ini memilih feature set memakai development CV, bukan final test.

Secara default command ini memakai dua anchor:
- `Logistic Regression` untuk melihat pola linear/sederhana
- `Random Forest` untuk melihat pola nonlinear

Random Forest di sini hanya alat bantu sensitivitas feature set. Ia tidak masuk ke tabel perbandingan 6 model utama.

Feature set kandidat:
- `clinical_core`
- `clinical_core_extreme`
- `clinical_core_extreme_weight`
- `full_transcript_comparator`

Output utama:
- `outputs/app/train/feature_set_screening_summary.csv`
- `outputs/app/train/feature_set_screening_anchor_summary.csv`
- `outputs/app/train/feature_set_screening_report.md`
- `outputs/app/train/selected_feature_set.json`

### 5. Cross-Validation Model

```powershell
python main.py cv-main
```

Membandingkan 6 algoritma pada feature set yang sama.

Model yang dibandingkan:
- Logistic Regression
- SVM
- KNN
- Gradient Boosting
- XGBoost
- LightGBM

Output utama:
- `outputs/app/train/main_cv_summary.csv`
- `outputs/app/train/main_cv_report.md`

### 6. Imbalance Experiments

```powershell
python main.py weight-compare
python main.py smote-compare
```

`weight-compare` membandingkan weighted vs unweighted.

`smote-compare` membandingkan tanpa SMOTE vs dengan SMOTE.

SMOTE hanya diterapkan pada training fold, bukan pada validation/test.

### 7. Optuna

```powershell
python main.py optuna-tune
```

Melakukan tuning hyperparameter pada development CV dengan metric utama `PR-AUC`.

Output utama:
- `outputs/app/train/optuna_trials.csv`
- `outputs/app/train/optuna_best_params.json`
- `outputs/app/train/optuna_baseline_vs_tuned.csv`
- `outputs/app/train/optuna_report.md`

### 8. Threshold Tuning

```powershell
python main.py threshold-tune
```

Mencari threshold klasifikasi dari out-of-fold prediction di development set.

Output utama:
- `outputs/app/train/threshold_tuning_results.csv`
- `outputs/app/train/threshold_tuning_best_thresholds.csv`
- `outputs/app/train/threshold_tuning_report.md`

### 9. Final Evaluation

```powershell
python main.py train
```

Melatih model final pada seluruh development data dan mengevaluasi sekali pada locked test.

Output utama:
- `outputs/app/train/final_test_results.csv`
- `outputs/app/train/final_test_predictions.csv`
- `outputs/app/train/report.md`
- `outputs/app/train/best_model.pkl`
- `outputs/app/train/calibration_*.png`

### 10. Error Analysis

```powershell
python main.py error-analysis
```

Membaca pola false positive dan false negative pada locked test.

Output utama:
- `outputs/app/train/error_analysis_summary.csv`
- `outputs/app/train/error_analysis_predictions.csv`
- `outputs/app/train/error_analysis_report.md`

### 11. XAI / SHAP

```powershell
python main.py xai
```

Menjalankan interpretasi SHAP pada model final.

Output utama:
- `outputs/app/train/xai/xai_report.md`
- `outputs/app/train/xai/shap_summary_plot.png`
- `outputs/app/train/xai/shap_summary_bar.png`
- `outputs/app/train/xai/xai_force_*.html`

## Jalankan Semua Tahap

```powershell
python main.py all
```

Urutan yang dijalankan:

1. prepare data
2. EDA
3. methodology check
4. feature set screening
5. cross-validation model
6. leakage audit/check
7. weighting comparison
8. SMOTE comparison
9. Optuna
10. threshold tuning
11. final evaluation
12. error analysis
13. XAI

## Catatan Penting

Model utama sengaja tidak memakai:
- `State`
- `PhySp_*`
- diagnosis / ICD
- medication
- `PatientGuid`
- `PracticeGuid`

Alasannya bukan karena kolom itu selalu tidak berguna, tetapi karena untuk model utama kolom tersebut berisiko menjadi shortcut terhadap lokasi, jalur layanan, diagnosis yang sudah jadi, obat, atau identitas fasilitas.
