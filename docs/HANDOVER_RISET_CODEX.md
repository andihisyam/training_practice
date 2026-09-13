# Handover Riset PracticeFusion T2DM

Dokumen ini dibuat sebagai pegangan jika penelitian dilanjutkan di laptop lain atau sesi Codex lain. Baca dokumen ini dulu sebelum mengubah kode, menjalankan ulang eksperimen, atau menulis laporan.

## 1. Inti Penelitian

Penelitian ini membangun model machine learning untuk **klasifikasi status Diabetes Mellitus Tipe 2 (T2DM)** berbasis data Electronic Health Record Practice Fusion.

Target model:

```text
DMIndicator
0 = pasien non-DM
1 = pasien DM
```

Posisi metodologi final:

- Ini adalah **klasifikasi status DM vs non-DM berbasis EHR**, bukan prediksi pasien akan terkena diabetes di masa depan.
- Alasannya: dataset yang tersedia tidak memberikan tanggal diagnosis T2DM pertama yang cukup jelas untuk membangun observation window dan prediction window.
- Konteks penggunaan model lebih dekat ke **screening/identifikasi risiko/status**, sehingga recall penting, tetapi model tetap dievaluasi dengan PR-AUC agar tidak hanya mengejar recall tinggi yang tidak berguna.

## 2. Repo dan File Penting

File/folder utama:

- `main.py`: entrypoint utama. Bisa dipakai sebagai CLI atau menu interaktif.
- `practicefusion/pipelines/prepare_data.py`: membangun final dataset.
- `practicefusion/pipelines/eda.py`: EDA.
- `practicefusion/pipelines/train.py`: training, CV, imbalance, Optuna, threshold, error analysis, XAI.
- `practicefusion/interactive_menu.py`: menu interaktif saat menjalankan `python main.py`.
- `docs/README_PENGGUNAAN_MAIN.md`: panduan command.
- `docs/README_EKSPERIMEN_DAN_HASIL.md`: ringkasan eksperimen dan hasil.
- `final_outputs/`: artefak final yang sengaja dipilih untuk dipush ke GitHub.
- `outputs/`: hasil run lokal yang lengkap, tetapi di-ignore oleh Git.

Catatan:

- `outputs/` tidak dipush karena berisi banyak file besar dan file antara.
- `final_outputs/` dibuat sebagai versi ringkas dan bersih dari output penting untuk GitHub.

## 3. Keputusan Metodologi Besar

### 3.1 Kolom yang Tidak Dipakai di Model Utama

Model utama sengaja tidak memakai:

- `State`
- `PatientGuid`
- `PracticeGuid`
- `PhySp_*`
- diagnosis / ICD
- medication

Alasan:

- `PatientGuid` dan `PracticeGuid` adalah identitas/metadata, bukan karakteristik klinis.
- `State` bisa menjadi proxy lokasi, fasilitas, atau pola pencatatan.
- `Physician Specialty` bisa menjadi proxy jalur layanan setelah pasien sudah diketahui sakit.
- ICD/diagnosis terlalu dekat dengan label DM.
- Medication bisa langsung menunjukkan pasien sedang diterapi/monitoring diabetes.

Keputusan penting:

- Kolom ini bukan berarti "tidak berguna secara statistik".
- Kolom ini sengaja dikeluarkan agar model final lebih defensible secara metodologi.
- Diagnosis/ICD dan medication tetap dipakai untuk leakage audit/check, bukan predictor final.

### 3.2 Kenapa ICD Dibuang

ICD dianggap berisiko leakage karena pada pasien baru, belum tentu ICD sudah tersedia sebelum keputusan screening dibuat. Kalau model memakai ICD, model bisa sekadar membaca kode diagnosis yang sudah dekat dengan outcome.

Argumen sidang:

> Model utama tidak memakai ICD karena penelitian ingin membangun klasifikasi berbasis fitur klinis dasar yang lebih mungkin tersedia sebelum atau tanpa diagnosis formal diabetes.

### 3.3 Kenapa Medication Dibuang

Medication juga dibuang dari model utama karena obat bisa sangat dekat dengan status DM. Misalnya insulin/metformin dapat langsung menjadi sinyal pasien diabetes.

Argumen sidang:

> Medication berpotensi membuat model belajar shortcut, bukan pola klinis dasar pasien.

### 3.4 Missing Value dan Imputation Flag

Pendekatan preprocessing:

- Nilai vital/agregat yang kosong diimputasi.
- Jika suatu kolom diimputasi, dibuat flag indikator missing/imputed agar model tahu bahwa nilai tersebut hasil imputasi.
- Untuk kolom transcript yang bernilai `0`, keputusan finalnya tidak semua `0` otomatis dianggap missing.

Keputusan tentang `0`:

- Untuk kolom agregat seperti `Min`, `Mean`, `Max`, `Change`, dan `NObs`, nilai `0` tidak langsung dianggap salah.
- Alasannya: dalam data hasil agregasi, `0` bisa bermakna tidak ada observasi atau nilai agregat tertentu tergantung konteks kolom.
- Kolom `std` akhirnya tidak dipakai karena kurang stabil dan lebih sulit dijelaskan dalam konteks data sparse.

## 4. Dataset yang Dipakai

Awalnya data lama dipakai sebagai basis, lalu fitur yang kurang informatif direbuild dari data raw `d1` sampai `d5`.

Keputusan final:

- Dataset final dibangun ulang di tahap `prepare-data`.
- Transcript direbuild lebih lengkap dari data raw.
- Gender diambil dari data pasien yang lebih cocok.
- Heart rate dibuang karena tidak tersedia/kurang bernilai.
- Medication missing tidak ditebak. Jika obat pasien tidak tersedia, tidak dilakukan imputasi obat untuk model final.

Output final dataset lokal:

```text
outputs/app/prepare_data/final_dataset.csv
```

File ini tidak dimasukkan ke `final_outputs/` karena ukurannya besar.

## 5. Feature Set Final

Feature set dibuat eksplisit dengan allow-list.

### Set A - Clinical Core

```text
Age
Gender
BMI_Mean
SystolicBP_Mean
DiastolicBP_Mean
```

Tujuan: baseline klinis sederhana.

### Set B - Clinical Core + Extreme

```text
Age
Gender
BMI_Mean
BMI_Max
SystolicBP_Mean
SystolicBP_Max
DiastolicBP_Mean
DiastolicBP_Max
```

Tujuan: kandidat utama. Tetap sederhana, tetapi menangkap nilai rata-rata dan nilai ekstrem yang mungkin informatif.

### Set C - Core + Weight

Set B ditambah:

```text
Weight_Mean
Weight_Max
```

Tujuan: mengecek apakah berat badan menambah informasi setelah BMI tersedia.

### Set D - Full Transcript Comparator

Memakai fitur transcript lebih lengkap, tetapi tetap tanpa State, physician, diagnosis, medication, dan ID.

Tujuan: pembanding kompleks. Bukan otomatis kandidat utama.

## 6. Strategi Evaluasi

Split data:

- 70% development data
- 30% locked final test

Semua keputusan dilakukan di development data:

- feature set selection
- model comparison
- imbalance strategy
- Optuna tuning
- threshold tuning

Locked test hanya dipakai setelah keputusan final ditetapkan.

Ini penting karena test set tidak boleh dipakai berkali-kali untuk memilih metode.

## 7. Eksperimen yang Sudah Dilakukan

### 7.1 Prepare Data

Command:

```powershell
python main.py prepare-data
```

Tujuan:

- membangun final dataset
- rebuild transcript dari raw data
- menghasilkan report prepare data

Output penting:

- `outputs/app/prepare_data/final_dataset.csv`
- `outputs/app/prepare_data/prepare_report.md`
- `final_outputs/01_prepare_data/prepare_report.md`

### 7.2 EDA

Command:

```powershell
python main.py eda
```

Tujuan:

- melihat distribusi label
- melihat missing value
- melihat ringkasan fitur numerik
- melihat korelasi dan boxplot fitur penting

Output penting:

- `final_outputs/02_eda/report.md`
- `final_outputs/02_eda/eda_summary.json`
- `final_outputs/02_eda/plots/`

### 7.3 Methodology Check

Command:

```powershell
python main.py methodology-check
```

Yang dicek:

- satu pasien hanya satu baris
- development/test tidak bercampur
- split reproducible
- CV tidak menyentuh locked test
- feature set utama tidak memakai fitur terlarang
- Set B sesuai allow-list eksplisit
- SMOTE hanya didesain pada training fold

Hasil terakhir:

```text
overall_status = pass
```

Output:

- `final_outputs/03_methodology/methodology_checks_report.md`

### 7.4 Leakage Audit dan Leakage Check

Tujuan:

- membuktikan kenapa diagnosis/ICD, medication, physician, dan sejenisnya tidak dipakai di model utama.

Hasil leakage check terbaru dengan XGBoost:

- clean transcript comparator PR-AUC: `0.4416`
- dengan diagnosis PR-AUC: `0.4820`
- dengan medication PR-AUC: `0.4753`
- full comparator PR-AUC: `0.5024`
- delta full vs clean: `+0.0608`

Interpretasi:

- Ada kenaikan performa saat diagnosis/medication dimasukkan.
- Kenaikan ini mendukung dugaan adanya shortcut/leakage.
- Karena itu diagnosis dan medication tidak dipakai di primary model.

### 7.5 Feature Set Screening

Command:

```powershell
python main.py feature-set-screen
```

Default terbaru:

- anchor 1: `Logistic Regression`
- anchor 2: `Random Forest`

Random Forest hanya dipakai sebagai sensitivity check untuk pemilihan feature set. Random Forest **tidak** masuk ke 6 model utama proposal.

Hasil sensitivity dua anchor:

| Feature Set | Mean PR-AUC | Recall | Catatan |
|---|---:|---:|---|
| Set D - Full Transcript Comparator | `0.4225` | `0.4417` | paling tinggi, tetapi paling kompleks |
| Set C - Core + Weight | `0.4201` | `0.4718` | sedikit di bawah Set D |
| Set B - Clinical Core + Extreme | `0.4164` | `0.4884` | dipilih karena lebih sederhana dan masih kompetitif |
| Set A - Clinical Core | `0.3948` | `0.4917` | terlalu sederhana, PR-AUC lebih rendah |

Keputusan:

```text
Selected feature set = clinical_core_extreme
```

Alasan:

- Set D memang PR-AUC tertinggi, tetapi jauh lebih kompleks.
- Set B masih kompetitif dan lebih mudah dijelaskan.
- Set B tidak memakai fitur penuh transcript yang berpotensi lebih sulit dipertanggungjawabkan.

### 7.6 Main Cross-Validation Model

Command:

```powershell
python main.py cv-main
```

Model yang dibandingkan:

- Logistic Regression
- SVM
- KNN
- Gradient Boosting
- XGBoost
- LightGBM

Hasil pada `clinical_core_extreme`:

| Model | Mean PR-AUC | Recall | F2 |
|---|---:|---:|---:|
| XGBoost | `0.4393` | `0.7194` | `0.5886` |
| Gradient Boosting | `0.4269` | `0.7276` | `0.5915` |
| LightGBM | `0.4216` | `0.6576` | `0.5605` |
| Logistic Regression | `0.4175` | `0.7103` | `0.5882` |
| SVM | `0.4153` | `0.7427` | `0.6031` |
| KNN | `0.3756` | `0.1798` | `0.2045` |

Keputusan:

```text
Model utama = XGBoost
```

Alasan:

- PR-AUC tertinggi.
- Recall dan F2 tetap cukup baik.
- Cocok dengan kebutuhan ranking/screening pada data imbalance.

### 7.7 Imbalance Experiment

Command:

```powershell
python main.py weight-compare
python main.py smote-compare
```

Eksperimen:

- unweighted model
- weighted model
- SMOTE

Hasil utama XGBoost:

| Strategi | PR-AUC | Recall | F2 |
|---|---:|---:|---:|
| Weighted XGBoost | `0.4393` | `0.7194` | `0.5886` |
| Unweighted XGBoost | `0.4366` | `0.1475` | `0.1727` |
| SMOTE XGBoost | `0.4231` | `0.5711` | `0.5210` |

Keputusan:

```text
Strategi utama = class weighting
```

Alasan:

- Unweighted model recall sangat rendah.
- SMOTE meningkatkan recall dibanding unweighted, tetapi PR-AUC turun dan masih kalah dari weighted model.
- Pada data sparse/agregat seperti ini, SMOTE tidak otomatis lebih baik.

### 7.8 Optuna

Command:

```powershell
python main.py optuna-tune
```

Tujuan:

- tuning hyperparameter XGBoost pada development CV
- objective utama: PR-AUC

Hasil:

- baseline XGBoost PR-AUC: `0.4393`
- tuned XGBoost PR-AUC: `0.4423`
- delta: `+0.0030`
- tuned recall: `0.7299`

Interpretasi:

- Optuna memberi peningkatan kecil.
- Tetap layak dilaporkan karena tuning dilakukan terkontrol pada development CV.

### 7.9 Threshold Tuning

Command:

```powershell
python main.py threshold-tune
```

Threshold dipilih dari out-of-fold prediction pada development data, bukan dari locked test.

Hasil:

- threshold terpilih: `0.35`
- recall development OOF: `0.8781`
- precision development OOF: `0.2919`
- specificity development OOF: `0.4955`
- F2 development OOF: `0.6265`

Keputusan:

```text
Final threshold = 0.35
```

Alasan:

- Dalam konteks screening medis, false negative lebih berbahaya daripada false positive.
- Threshold 0.35 menaikkan recall dengan trade-off specificity/precision yang masih bisa dijelaskan.

### 7.10 Final Evaluation

Command:

```powershell
python main.py train
```

Konfigurasi final:

- feature set: `clinical_core_extreme`
- model: `XGBoost`
- imbalance handling: weighted
- hyperparameter: Optuna tuned
- threshold: `0.35`

Hasil locked test:

| Metric | Nilai |
|---|---:|
| PR-AUC | `0.4222` |
| ROC-AUC | `0.7678` |
| Recall / Sensitivity | `0.8737` |
| Specificity | `0.4915` |
| Precision | `0.2894` |
| NPV | `0.9426` |
| F1 | `0.4347` |
| F2 | `0.6223` |
| Brier Score | `0.1927` |

Confusion matrix:

|  | Pred 0 | Pred 1 |
|---|---:|---:|
| Actual 0 | TN `1182` | FP `1223` |
| Actual 1 | FN `72` | TP `498` |

Interpretasi:

- Model berhasil menangkap mayoritas pasien DM.
- False positive cukup tinggi, tetapi ini masih sejalan dengan positioning screening.
- NPV tinggi menunjukkan pasien yang diprediksi non-DM relatif aman.
- Precision rendah perlu dijelaskan karena label DM imbalance dan threshold sengaja dibuat lebih sensitif.

### 7.11 Error Analysis

Command:

```powershell
python main.py error-analysis
```

Tujuan:

- melihat pola false positive dan false negative
- membantu menjelaskan kekuatan dan kelemahan model

Output:

- `final_outputs/09_error_analysis/error_analysis_report.md`
- `final_outputs/09_error_analysis/error_analysis_summary.csv`
- `final_outputs/09_error_analysis/error_analysis_numeric_profile.csv`

### 7.12 XAI / SHAP

Command:

```powershell
python main.py xai
```

Output penting:

- `final_outputs/10_xai/xai_report.md`
- `final_outputs/10_xai/xai_top_features.csv`
- `final_outputs/10_xai/plots/shap_summary_plot.png`
- `final_outputs/10_xai/plots/shap_summary_bar.png`
- `final_outputs/10_xai/plots/shap_dependence_*.png`
- `final_outputs/10_xai/force_plots/xai_force_*.html`

Top fitur SHAP final:

- `Age`
- `BMI_Max`
- `DiastolicBP_Mean`
- `BMI_Mean`
- `SystolicBP_Max`
- `Gender_F`
- `SystolicBP_Mean`
- `DiastolicBP_Max`
- `Gender_M`

Interpretasi utama:

- Model banyak dipengaruhi umur, BMI, dan tekanan darah.
- Ini masuk akal secara klinis dan lebih defensible daripada model yang bergantung pada ICD atau medication.
- SHAP hanya menjelaskan perilaku model, bukan membuktikan kausalitas medis.

## 8. Kesimpulan Final Saat Ini

Kesimpulan metodologi:

- Model final sebaiknya memakai fitur klinis dasar, bukan ICD/medication/physician/state.
- `clinical_core_extreme` adalah feature set utama yang paling seimbang antara performa dan kesederhanaan.
- XGBoost weighted adalah model utama.
- SMOTE sudah diuji, tetapi tidak menjadi strategi utama.
- Optuna memberikan peningkatan kecil dan tetap dipakai.
- Threshold `0.35` dipakai karena konteks screening lebih mengutamakan recall.
- XAI/SHAP sudah tersedia untuk interpretasi global dan lokal.

Kesimpulan hasil:

```text
Best final configuration:
Feature set  = clinical_core_extreme
Model        = XGBoost
Imbalance    = class weighting
Tuning       = Optuna
Threshold    = 0.35
Final recall = 0.8737
Final PR-AUC = 0.4222
Final ROC-AUC = 0.7678
```

## 9. Cara Menjalankan di Laptop Lain

Urutan minimal untuk memahami hasil tanpa rerun semua:

```powershell
python main.py
```

Lalu pilih menu:

```text
2. Lihat Ringkasan Hasil
3. Buka Visualisasi / Report
4. Kesimpulan Umum
```

Urutan rerun lengkap:

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

Atau:

```powershell
python main.py all
```

Catatan:

- Runtime bisa lama, terutama CV, Optuna, dan XAI.
- Pastikan dependency Python sudah lengkap.
- Jangan mengubah locked test split jika ingin hasil tetap comparable.

## 10. File Output Final yang Layak Dipakai di Laporan

Gunakan artefak dari `final_outputs/`, bukan langsung dari `outputs/`.

File paling penting:

- `final_outputs/99_summary/general_conclusion.md`
- `final_outputs/99_summary/rebuild_research_summary.md`
- `final_outputs/04_feature_set_screening/feature_set_screening_report.md`
- `final_outputs/05_model_comparison/main_cv_summary.csv`
- `final_outputs/06_imbalance/weighting_comparison_summary.csv`
- `final_outputs/06_imbalance/smote_comparison_summary.csv`
- `final_outputs/07_optuna_threshold/optuna_baseline_vs_tuned.csv`
- `final_outputs/07_optuna_threshold/threshold_tuning_best_thresholds.csv`
- `final_outputs/08_final_evaluation/final_test_results.csv`
- `final_outputs/08_final_evaluation/calibration_clinical_core_extreme_XGBoost.png`
- `final_outputs/09_error_analysis/error_analysis_report.md`
- `final_outputs/10_xai/xai_report.md`
- `final_outputs/10_xai/plots/shap_summary_plot.png`
- `final_outputs/10_xai/plots/shap_summary_bar.png`

## 11. Hal yang Jangan Diubah Sembarangan

Jangan langsung mengubah ini tanpa alasan metodologi kuat:

- target `DMIndicator`
- locked test split `random_state=42`
- penghapusan ICD/diagnosis dari primary model
- penghapusan medication dari primary model
- feature set final `clinical_core_extreme`
- final threshold `0.35`
- posisi penelitian sebagai classification, bukan future prediction

Kalau berubah, semua narasi metodologi dan sebagian hasil harus ditulis ulang.

## 12. Hal yang Masih Bisa Dilanjutkan

Yang masih bisa dilakukan jika ada waktu:

- memperbaiki visualisasi laporan agar lebih siap masuk skripsi
- membuat tabel final dalam format Word/LaTeX
- menulis bagian metodologi skripsi dari dokumen ini
- membuat slide sidang dari `final_outputs/`
- mengecek dependency/requirements agar setup laptop lain lebih mudah
- menambah external validation jika ada dataset lain
- membuat sensitivity analysis tambahan, tetapi jangan memakai locked test untuk memilih ulang keputusan

## 13. Jawaban Singkat Kalau Ditanya Dosen

### Kenapa bukan future prediction?

Karena tidak tersedia tanggal diagnosis T2DM pertama yang jelas untuk membangun prediction window. Jadi penelitian diposisikan sebagai klasifikasi status DM/non-DM berbasis EHR.

### Kenapa ICD dan obat dibuang?

Karena keduanya terlalu dekat dengan label DM. Jika dipakai, model bisa belajar shortcut, bukan pola klinis dasar pasien.

### Kenapa pakai Set B?

Karena Set B memberikan performa kompetitif dibanding set yang lebih kompleks, tetapi lebih sederhana, lebih klinis, dan lebih mudah dijelaskan.

### Kenapa SMOTE tidak dipakai?

SMOTE sudah diuji, tetapi pada data ini performanya kalah dari class weighting, terutama dari sisi PR-AUC dan F2.

### Kenapa threshold 0.35?

Karena konteks screening lebih mengutamakan recall. Threshold 0.35 menurunkan false negative, walaupun false positive meningkat.

### Apa makna SHAP?

SHAP menunjukkan fitur apa yang paling memengaruhi output model. Pada model final, fitur pentingnya adalah umur, BMI, dan tekanan darah. SHAP bukan bukti sebab-akibat.

## 14. Status Git Terakhir yang Diinginkan

Branch final sebaiknya:

```text
main
```

Folder yang sebaiknya dipush:

- kode Python
- `docs/`
- `notebooks/`
- `final_outputs/`
- `README.md`
- `.gitignore`

Folder yang tidak perlu dipush:

- `outputs/`
- cache Python
- virtual environment
- log lokal

Sebelum push, cek:

```powershell
git status --short
git add docs/HANDOVER_RISET_CODEX.md final_outputs
git commit -m "Add research handover and final outputs"
git push origin main
```
