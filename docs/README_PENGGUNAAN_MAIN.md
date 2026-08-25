# Panduan Sederhana `python main.py`

Dokumen ini menjelaskan cara memakai `main.py` dengan bahasa sederhana.

Tujuan utamanya:
- memudahkan dosen atau penguji menjalankan pipeline tanpa harus hafal semua command
- memudahkan melihat hasil tanpa membuka folder satu per satu

---

## 1. Fungsi `main.py` itu apa?

`main.py` adalah pintu utama untuk seluruh alur penelitian ini.

Dari file ini, pengguna bisa:
- menyiapkan dataset final
- menjalankan EDA
- melatih model
- menjalankan eksperimen tambahan seperti leakage check, weighting, SMOTE, cross-validation, threshold tuning, error analysis, dan XAI
- melihat hasil report dan visualisasi dari menu interaktif

---

## 2. Cara paling mudah: mode menu interaktif

Jalankan:

```powershell
python main.py
```

Kalau command di atas dijalankan tanpa argumen tambahan, akan muncul menu interaktif.

Menu utama berisi:

1. `Jalankan Pipeline / Eksperimen`
2. `Lihat Ringkasan Hasil`
3. `Buka Visualisasi / Report`
4. `Kesimpulan Umum`
0. `Keluar`

### Kegunaan tiap menu

#### 1. Jalankan Pipeline / Eksperimen
Dipakai untuk menjalankan tahap analisis.

Submenu ini berisi:
- `Prepare Data`
- `EDA`
- `Baseline Training`
- `Leakage Audit`
- `Leakage Check`
- `Weighting Comparison`
- `SMOTE Comparison`
- `Cross Validation`
- `Threshold Tuning`
- `Error Analysis`
- `XAI`
- `Jalankan Semua Tahap Bersih`

Artinya, pengguna tidak perlu hafal command satu per satu.

#### 2. Lihat Ringkasan Hasil
Dipakai untuk membaca report teks langsung dari terminal.

Submenu ini dipakai untuk:
- membaca report prepare data
- membaca report EDA
- membaca report training
- membaca report leakage
- membaca report weighting
- membaca report SMOTE
- membaca report cross-validation
- membaca report threshold tuning
- membaca report error analysis
- membaca report XAI
- melihat top fitur SHAP
- melihat status file hasil

#### 3. Buka Visualisasi / Report
Dipakai untuk membuka file hasil secara langsung, misalnya:
- SHAP summary plot
- SHAP bar plot
- SHAP dependence plot
- force plot pasien individual
- report XAI

Ini berguna supaya pengguna tidak perlu masuk ke folder `outputs/` secara manual.

#### 4. Kesimpulan Umum
Dipakai untuk melihat ringkasan akhir penelitian dalam satu layar:
- model terbaik
- feature set yang dipakai
- alasan pemilihan feature set
- hasil holdout
- hasil cross-validation
- threshold final
- ringkasan XAI

Menu ini cocok dipakai saat demo atau sidang.

---

## 3. Cara kedua: mode command biasa

Kalau pengguna ingin langsung menjalankan command tertentu, `main.py` juga tetap mendukung mode CLI biasa.

Contoh:

### Prepare data

```powershell
python main.py prepare-data
```

Output utama:
- `outputs/app/prepare_data/final_dataset.csv`
- `outputs/app/prepare_data/transcript_rebuilt.csv`
- `outputs/app/prepare_data/prepare_report.md`

### EDA

```powershell
python main.py eda
```

Output utama:
- `outputs/app/eda/report.md`
- `outputs/app/eda/eda_summary.json`

### Baseline training

```powershell
python main.py train
```

Output utama:
- `outputs/app/train/baseline_results.csv`
- `outputs/app/train/report.md`
- `outputs/app/train/best_model.pkl`

### Leakage audit

```powershell
python main.py leakage-audit
```

### Leakage check

```powershell
python main.py leakage-check
```

### Weighted vs unweighted

```powershell
python main.py weight-compare
```

### SMOTE comparison

```powershell
python main.py smote-compare
```

### Cross-validation utama

```powershell
python main.py cv-main
```

### Threshold tuning

```powershell
python main.py threshold-tune
```

### Error analysis

```powershell
python main.py error-analysis
```

### XAI / SHAP

```powershell
python main.py xai
```

---

## 4. Kapan pakai menu, kapan pakai command?

### Pakai menu interaktif kalau:
- ingin demo ke dosen
- ingin lihat hasil dengan cepat
- tidak ingin menghafal command

### Pakai command biasa kalau:
- ingin eksperimen teknis lebih cepat
- ingin otomasi
- ingin menjalankan tahap tertentu saja dari terminal

---

## 5. Jalur paling praktis untuk dosen

Kalau dosen hanya ingin melihat alur penelitian secara umum, langkah termudah adalah:

1. Jalankan:

```powershell
python main.py
```

2. Pilih:
- `4` untuk melihat **Kesimpulan Umum**
- `2` untuk membaca report hasil
- `3` untuk membuka gambar SHAP

Dengan alur ini, dosen tidak perlu membuka folder output secara manual.

---

## 6. File hasil penting yang biasanya dilihat

### Report teks
- `outputs/app/prepare_data/prepare_report.md`
- `outputs/app/eda/report.md`
- `outputs/app/train/report.md`
- `outputs/app/train/main_cv_report.md`
- `outputs/app/train/error_analysis_report.md`
- `outputs/app/train/xai/xai_report.md`
- `outputs/app/train/general_conclusion.md`

### Visualisasi SHAP
- `outputs/app/train/xai/shap_summary_plot.png`
- `outputs/app/train/xai/shap_summary_bar.png`
- `outputs/app/train/xai/shap_dependence_Age.png`
- `outputs/app/train/xai/shap_dependence_BMI_Mean.png`
- `outputs/app/train/xai/shap_dependence_BMI_Max.png`

---

## 7. Catatan penting

- Model utama penelitian saat ini **tidak memakai ICD sebagai feature utama**
- diagnosis dan medication tetap dipakai sebagai pembanding/sensitivitas, bukan fondasi model final
- threshold final model utama saat ini adalah `0.50`
- SHAP dipakai untuk interpretasi global dan lokal

---

## 8. Jika ingin menjalankan seluruh pipeline bersih sekaligus

Ada dua cara:

### Lewat menu interaktif
Pilih:
- `Jalankan Pipeline / Eksperimen`
- lalu pilih `Jalankan Semua Tahap Bersih`

### Lewat script

```powershell
python scripts/run_clean_pipeline_end_to_end.py
```

Script ini akan menjalankan:
- prepare data
- EDA
- training
- leakage audit
- leakage check
- weighting comparison
- SMOTE comparison
- cross-validation
- threshold tuning
- error analysis
- XAI

---

Dokumen ini dibuat agar penggunaan `main.py` mudah dipahami oleh pengguna non-teknis, termasuk dosen pembimbing dan penguji.
