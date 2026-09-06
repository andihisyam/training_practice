# PracticeFusion T2DM Classification

Workspace ini berisi pipeline penelitian untuk **klasifikasi status Diabetes Mellitus Tipe 2 (T2DM)** berbasis data Electronic Health Record (EHR).

Penelitian ini tidak diklaim sebagai prediksi kejadian diabetes di masa depan, karena dataset tidak menyediakan tanggal diagnosis T2DM pertama yang cukup jelas untuk membentuk observation window dan prediction window.

## Dokumen Penting

- [Panduan penggunaan `main.py`](docs/README_PENGGUNAAN_MAIN.md)
- [Ringkasan eksperimen dan hasil penelitian](docs/README_EKSPERIMEN_DAN_HASIL.md)

## Struktur Folder

- `data/raw/`: dataset mentah CSV
- `docs/`: proposal, rencana, dan dokumentasi penelitian
- `notebooks/`: notebook eksplorasi
- `outputs/`: hasil EDA, training, dan XAI
- `scripts/`: script utilitas tambahan
- `practicefusion/`: modul Python utama
- `main.py`: entrypoint CLI dan menu interaktif

## Cara Menjalankan

Mode menu interaktif:

```powershell
python main.py
```

Mode command:

```powershell
python main.py prepare-data
python main.py eda
python main.py methodology-check
python main.py feature-set-screen
python main.py cv-main
python main.py weight-compare
python main.py smote-compare
python main.py optuna-tune
python main.py threshold-tune
python main.py train
python main.py error-analysis
python main.py xai
```

Atau jalankan alur lengkap:

```powershell
python main.py all
```

## Feature Set Utama

Feature set kandidat setelah refactor:

- `clinical_core`: Age, Gender, BMI mean, systolic BP mean, diastolic BP mean
- `clinical_core_extreme`: `clinical_core` + BMI max, systolic BP max, diastolic BP max
- `clinical_core_extreme_weight`: `clinical_core_extreme` + Weight mean/max
- `full_transcript_comparator`: transcript-derived features yang lebih lengkap

`feature-set-screen` secara default memakai dua anchor, yaitu `Logistic Regression` dan `Random Forest`. Random Forest hanya dipakai untuk menguji stabilitas pemilihan feature set, bukan sebagai model utama dalam perbandingan 6 algoritma proposal.

Model utama sengaja tidak memakai:

- `State`
- `PhySp_*`
- diagnosis / ICD
- medication
- `PatientGuid`
- `PracticeGuid`

Kolom tersebut disimpan sebagai metadata atau bahan audit/sensitivity analysis, tetapi bukan predictor primary model.

## Prinsip Evaluasi

Semua keputusan dilakukan pada 70% development data:

- feature set selection
- model comparison
- imbalance strategy comparison
- Optuna tuning
- threshold tuning

Setelah keputusan selesai, model final dievaluasi satu kali pada 30% locked test.

Primary metric untuk model selection adalah `PR-AUC`. Recall tetap penting karena konteks penelitian adalah screening/identifikasi status T2DM.

## Hasil Refactor Terbaru

Feature set yang dipilih:

- `clinical_core_extreme`
- label: `Set B - Clinical Core + Extreme`
- alasan: pada screening dua anchor, performanya masih kompetitif terhadap feature set yang lebih kompleks, tetapi jauh lebih sederhana dan lebih mudah dijelaskan

Model final:

- XGBoost weighted
- Optuna tuned
- threshold final `0.35`

Hasil locked test:

- PR-AUC `0.4222`
- ROC-AUC `0.7678`
- Recall `0.8737`
- Specificity `0.4915`
- Precision `0.2894`
- F2 `0.6223`

Top fitur SHAP:

- Age
- BMI_Max
- DiastolicBP_Mean
- BMI_Mean
- SystolicBP_Max
