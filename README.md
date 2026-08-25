# PracticeFusion Workspace

Workspace ini sudah disiapkan untuk penelitian prediksi `Diabetes Mellitus Tipe 2 (T2DM)` berbasis data EHR.

## Struktur Folder

- `data/raw/`: dataset mentah CSV
- `docs/`: proposal dan catatan penelitian
- `notebooks/`: notebook eksplorasi / eksperimen
- `outputs/`: hasil EDA dan artefak analisis
- `scripts/`: script utilitas lama
- `practicefusion/`: modul Python utama untuk pipeline riset
- `main.py`: entrypoint CLI untuk menjalankan alur riset

## Ringkasan Data

- `patient.csv`: 9.948 baris x 5 kolom, berisi demografi dan label `DMIndicator`
- `diagnosis.csv`: 9.948 baris x 26 kolom, fitur diagnosis ICD-9 dan agregat kunjungan
- `physician_specialty.csv`: 9.948 baris x 62 kolom, frekuensi specialty dokter
- `transcript.csv`: 9.948 baris x 36 kolom, agregat tanda vital dan antropometri
- `medication.csv`: 9.836 baris x 2.368 kolom, fitur obat yang sangat high-dimensional

## Temuan Awal

- Distribusi label tidak seimbang: 1.904 pasien diabetes vs 8.044 non-diabetes.
- `medication.csv` tidak mencakup semua pasien, jadi join yang aman adalah `left join` dari `patient.csv`.
- Tidak ada duplikasi `PatientGuid` di semua tabel utama.
- Ada sinyal data khusus yang perlu hati-hati:
  - `BMI_Min = 0` muncul sangat sering dan tampak seperti nilai missing yang dikodekan sebagai nol.
  - `Gender` dan `State` di `patient.csv` saat ini sudah berbentuk angka, jadi perlu dicek lagi apakah memang sudah di-encode dari sumber aslinya.
  - Fitur diagnosis dan medication berpotensi leakage jika label diabetes dibentuk dari informasi yang sama.

## Cara Pakai

### 1. Siapkan final dataset

```powershell
python main.py prepare-data
```

Output default:
- `outputs/app/prepare_data/final_dataset.csv`
- `outputs/app/prepare_data/transcript_rebuilt.csv`

### 2. Jalankan EDA

```powershell
python main.py eda
```

Output default:
- `outputs/app/eda/report.md`
- `outputs/app/eda/eda_summary.json`

### 3. Jalankan baseline training

```powershell
python main.py train
```

Output default:
- `outputs/app/train/baseline_results.csv`
- `outputs/app/train/report.md`
- `outputs/app/train/best_model.pkl`
- `outputs/app/train/best_comparator_model.pkl` (jika feature set `full` ikut dijalankan)

Struktur feature set yang dipakai:
- `demografi_transcript`: model dasar paling bersih, berisi demografi + transcript
- `diagnosis`: model utama dengan tambahan diagnosis umum
- `diagnosis_physician`: model utama dengan tambahan diagnosis + specialty dokter
- `full`: model pembanding yang masih memakai medication untuk analisis leakage

### 4. Jalankan semua tahap sekaligus

```powershell
python main.py all
```

### 5. Contoh opsi tambahan

Jalankan hanya feature set tertentu:

```powershell
python main.py train --feature-sets demografi_transcript diagnosis
```

Jalankan hanya model tertentu:

```powershell
python main.py train --models "Logistic Regression" XGBoost LightGBM
```

### 6. Audit leakage fitur

```powershell
python main.py leakage-audit
```

Output default:
- `outputs/app/train/leakage_audit_report.md`
- `outputs/app/train/leakage_audit_keyword_hits.csv`
- `outputs/app/train/leakage_audit_high_association_medication.csv`
