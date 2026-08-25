# Ringkasan Eksperimen dan Hasil Penelitian

Dokumen ini merangkum eksperimen yang dilakukan dari awal sampai akhir, lengkap dengan perubahan keputusan metodologis yang diambil selama proses penelitian.

Tujuannya:
- menjadi catatan penelitian yang runtut
- membantu penulisan bab hasil dan pembahasan
- menjelaskan kenapa model final yang dipilih adalah model yang sekarang

---

## 1. Tujuan Penelitian

Penelitian ini bertujuan membandingkan beberapa model machine learning untuk prediksi `Diabetes Mellitus Tipe 2 (T2DM)` berbasis data EHR, lalu menginterpretasikan model terbaik menggunakan `SHAP`.

Model yang dibandingkan:
- Logistic Regression
- SVM
- KNN
- Gradient Boosting
- XGBoost
- LightGBM

---

## 2. Data yang Digunakan

Sumber data mentah:
- `d1.csv`
- `d2.csv`
- `d3.csv`
- `d4.csv`
- `d5.csv`
- `patient.csv`
- `diagnosis.csv`
- `medication.csv`
- `physician_specialty.csv`
- `transcript.csv`

Keputusan akhir yang dipakai pada pipeline final:
- label utama dan backbone pasien berasal dari `d5.csv`
- `Age` tetap diambil dari `patient.csv`
- transcript dibangun ulang dari `d2.csv`
- diagnosis, physician, dan medication dipertahankan sebagai blok agregat opsional

---

## 3. Tahap Prepare Data yang Dilakukan

Eksperimen dan keputusan di tahap data preparation:

### 3.1 Rebuild transcript
Transcript lama ditelaah ulang, lalu dibangun ulang dari `d2.csv`.

Fitur yang dibangun:
- `Min`
- `Max`
- `Mean`
- `NObs`
- `Change`

Untuk variabel:
- Height
- Weight
- BMI
- SystolicBP
- DiastolicBP
- RespiratoryRate
- Temperature

### 3.2 Cleaning nilai tidak masuk akal
Nilai `0` atau nilai di luar batas klinis diubah menjadi `NaN` untuk fitur transcript yang memang tidak logis jika bernilai nol.

Contoh:
- BMI
- tekanan darah
- respiratory rate
- temperature

### 3.3 Penghapusan fitur std
Seluruh fitur `*_Std` transcript akhirnya dibuang karena dianggap kurang stabil, lebih sulit dijelaskan, dan tidak terlalu mendukung tujuan model final.

### 3.4 Missing value strategy
Strategi akhir:
- kolom dengan missing terlalu tinggi dievaluasi
- missing indicator ditambahkan untuk fitur yang diimputasi
- beberapa kolom transcript tetap diimputasi median melalui pipeline model

### 3.5 Perubahan cohort final
Pada tahap awal, cohort final sempat bergantung pada irisan beberapa tabel sekaligus.

Keputusan final:
- cohort akhir dibentuk dari **backbone pasien + transcript rebuilt** sebagai syarat utama
- diagnosis, physician, dan medication digabung sebagai **optional left join**

Tujuan keputusan ini:
- menghindari pasien hilang hanya karena tidak punya medication atau blok agregat tertentu
- membuat dataset final lebih realistis dan lebih siap untuk model utama

### 3.6 Hasil akhir prepare data
- final dataset: `9916 x 2510`
- distribusi label:
  - non-DM: `8017`
  - DM: `1899`

Output utama:
- `outputs/app/prepare_data/final_dataset.csv`
- `outputs/app/prepare_data/prepare_report.md`

---

## 4. Tahap EDA yang Dilakukan

EDA dijalankan untuk:
- memeriksa distribusi label
- memeriksa kualitas data
- melihat pola umum fitur transcript, diagnosis, dan physician

Temuan utama:
- data bersifat tidak seimbang
- BMI, berat badan, tekanan darah, dan usia memberi sinyal klinis yang relevan
- diagnosis dan medication tampak memberi sinyal yang sangat dekat ke label

Output utama:
- `outputs/app/eda/report.md`
- `outputs/app/eda/eda_summary.json`

---

## 5. Struktur Feature Set yang Pernah Dicoba

Selama penelitian, beberapa rancangan feature set digunakan.

### Rancangan awal
- Demografi + Transcript
- Demografi + Transcript + Diagnosis
- Demografi + Transcript + Diagnosis + Physician
- Full + Medication

### Keputusan akhir
Feature set final dibagi menjadi:

#### Set A - Demografi + Transcript
Dipakai sebagai baseline paling bersih.

#### Set B - Demografi + Transcript + Physician
Dipakai sebagai model utama final.

#### Set C - Demografi + Transcript + Diagnosis
Dipakai sebagai **sensitivity set**, bukan model utama.

#### Set D - Demografi + Transcript + Diagnosis + Physician
Dipakai sebagai **sensitivity set**, bukan model utama.

#### Set E - Full + Medication
Dipakai sebagai **comparator leakage**.

---

## 6. Kenapa ICD Akhirnya Tidak Dipakai pada Model Utama

Ini adalah perubahan metodologis paling penting.

Pertimbangannya:
- ICD bisa membuat model terlalu menunggu diagnosis yang sudah jadi
- untuk pasien baru, ICD belum tentu tersedia
- dosen bisa mempertanyakan apakah model ini benar-benar prediksi dini atau hanya membaca diagnosis yang sudah terdokumentasi

Kesimpulan akhir:
- diagnosis **tidak dibuang dari eksperimen**
- tetapi diagnosis **diturunkan menjadi sensitivity feature set**
- model utama final tidak lagi bergantung pada ICD

---

## 7. Baseline Training yang Dilakukan

Semua model dibandingkan pada beberapa feature set.

### Hasil baseline final
Best primary holdout:
- model: `XGBoost`
- feature set: `Set B - Demografi + Transcript + Physician`
- PR-AUC: `0.4381`
- ROC-AUC: `0.7833`
- Recall: `0.7298`
- Precision: `0.3688`
- F1-score: `0.4900`

Best overall comparator:
- model: `LightGBM`
- feature set: `Set E - Full + Medication`
- PR-AUC: `0.4925`
- ROC-AUC: `0.8165`
- Recall: `0.6912`
- Precision: `0.4196`
- F1-score: `0.5222`

Makna hasil ini:
- model dengan medication memang lebih tinggi
- tetapi tidak dipakai sebagai model utama karena rawan leakage

---

## 8. Leakage Audit dan Leakage Check

Eksperimen leakage dilakukan dalam dua bentuk:

### 8.1 Leakage audit berbasis fitur
Tujuannya:
- mencari fitur diagnosis, physician, dan medication yang sangat dekat dengan label

Temuan:
- diagnosis meningkatkan sinyal model
- medication memberi sinyal yang lebih dekat lagi ke label
- beberapa token medication sangat berisiko leakage

### 8.2 Leakage check berbasis model
Tujuannya:
- mengukur kenaikan performa saat clean core diberi diagnosis atau medication

Ringkasan delta `full vs clean_core`:
- LightGBM: `+0.0550`
- Gradient Boosting: `+0.0448`
- XGBoost: `+0.0387`
- Logistic Regression: `-0.0249`

Interpretasi:
- medication dan diagnosis memang menambah performa
- tetapi kenaikan itu justru memperkuat alasan untuk memisahkan model utama dari feature yang terlalu dekat ke label

---

## 9. Penanganan Class Imbalance

Beberapa pendekatan diuji:

### 9.1 Weighted vs unweighted
Fokus pada model utama `Set B + XGBoost`:
- weighted PR-AUC: `0.4381`
- unweighted PR-AUC: `0.4357`
- weighted recall: `0.7298`
- unweighted recall: `0.1561`
- weighted F1: `0.4900`
- unweighted F1: `0.2465`

Keputusan:
- model weighted jauh lebih cocok
- unweighted terlalu rendah recall untuk konteks screening

### 9.2 SMOTE vs no SMOTE
Fokus pada model utama `Set B + XGBoost`:
- no SMOTE PR-AUC: `0.4381`
- SMOTE PR-AUC: `0.3950`
- no SMOTE recall: `0.7298`
- SMOTE recall: `0.8702`
- no SMOTE F1: `0.4900`
- SMOTE F1: `0.4509`

Keputusan:
- SMOTE menaikkan recall
- tetapi menurunkan PR-AUC dan F1
- SMOTE tidak dipakai sebagai default

---

## 10. Cross-Validation

Cross-validation utama dijalankan pada feature set primary final:
- Set A
- Set B

Hasil terbaik CV:
- model: `LightGBM`
- feature set: `Set B - Demografi + Transcript + Physician`
- mean PR-AUC: `0.4684 +/- 0.0167`
- mean recall: `0.6741 +/- 0.0209`
- mean precision: `0.3886 +/- 0.0109`
- mean F1: `0.4930 +/- 0.0140`

Model `XGBoost` tetap sangat kuat:
- mean PR-AUC: `0.4652 +/- 0.0159`
- mean recall: `0.7388 +/- 0.0218`
- mean F1: `0.4867 +/- 0.0126`

Interpretasi:
- XGBoost tetap unggul untuk recall
- LightGBM sedikit lebih unggul untuk kestabilan rata-rata PR-AUC dan F1

---

## 11. Threshold Tuning

Threshold tuning dilakukan untuk model kandidat utama.

Hasil akhir:
- `Set B + XGBoost`: threshold terbaik tetap `0.50`
- `Set A + LightGBM`: threshold terbaik `0.45`

Keputusan final untuk model utama:
- threshold `0.50`

Alasannya:
- lebih sesuai untuk konteks screening medis
- tidak menaikkan false negative secara berlebihan

---

## 12. Error Analysis

Error analysis dilakukan pada model final `Set B + XGBoost`.

### Threshold 0.50
- TP: `416`
- TN: `1693`
- FP: `712`
- FN: `154`

### Threshold 0.60
- TP: `306`
- TN: `1964`
- FP: `441`
- FN: `264`

Interpretasi:
- menaikkan threshold memang menurunkan false positive
- tetapi false negative naik banyak
- untuk konteks medis, threshold `0.50` lebih sesuai

Pola error yang ditemukan:
- false negative cenderung lebih muda, berat badan lebih rendah, BMI lebih rendah
- false positive cenderung lebih tua, berat badan lebih tinggi, BMI lebih tinggi

---

## 13. XAI dengan SHAP

XAI final dijalankan pada:
- model: `XGBoost`
- feature set: `Set B - Demografi + Transcript + Physician`
- threshold: `0.50`

Visualisasi yang disiapkan:
- `summary plot`
- `summary bar plot`
- `dependence plot`
- `force plot` pasien individual

Top fitur global SHAP:
1. `Age`
2. `BMI_Mean`
3. `BMI_Max`
4. `DiastolicBP_Mean`
5. `PhySp_Internal_Medicine`

Interpretasi umum:
- model membaca usia sebagai sinyal paling dominan
- BMI dan tekanan darah memberi kontribusi klinis yang kuat
- konteks layanan melalui specialty dokter tetap berpengaruh, tetapi tidak mendominasi secara tidak masuk akal

Keuntungan versi final ini:
- SHAP menjadi lebih mudah dipertahankan
- interpretasi model lebih klinis
- model tidak terlalu bergantung pada ICD atau medication

---

## 14. Model Final yang Dipilih

Untuk penulisan utama, posisi akhir penelitian ini adalah:

### Model utama
- `XGBoost`
- `Set B - Demografi + Transcript + Physician`
- threshold `0.50`
- weighted
- tanpa SMOTE

### Model pembanding kuat
- `LightGBM`
- `Set B - Demografi + Transcript + Physician`
- dipakai sebagai pembanding CV/stabilitas

### Model sensitivitas
- `Set C`
- `Set D`

### Comparator leakage
- `Set E - Full + Medication`

---

## 15. Artefak Hasil yang Tersedia

Beberapa file hasil penting:

### Prepare Data
- `outputs/app/prepare_data/prepare_report.md`

### EDA
- `outputs/app/eda/report.md`

### Training
- `outputs/app/train/report.md`

### Leakage
- `outputs/app/train/leakage_audit_report.md`
- `outputs/app/train/leakage_check_report.md`

### Imbalance
- `outputs/app/train/weighting_comparison_report.md`
- `outputs/app/train/smote_comparison_report.md`

### CV
- `outputs/app/train/main_cv_report.md`

### Threshold
- `outputs/app/train/threshold_tuning_report.md`

### Error Analysis
- `outputs/app/train/error_analysis_report.md`

### XAI
- `outputs/app/train/xai/xai_report.md`
- `outputs/app/train/xai/shap_summary_plot.png`
- `outputs/app/train/xai/shap_summary_bar.png`
- `outputs/app/train/xai/shap_dependence_Age.png`
- `outputs/app/train/xai/shap_dependence_BMI_Mean.png`
- `outputs/app/train/xai/shap_dependence_BMI_Max.png`

---

## 16. Kesimpulan Proses Penelitian

Secara keseluruhan, eksperimen penelitian bergerak dari:
- membangun ulang transcript
- membersihkan data dan missing values
- membandingkan beberapa model dan beberapa feature set
- mengevaluasi risiko leakage
- menguji strategi imbalance
- menguji kestabilan model melalui cross-validation
- memilih threshold yang sesuai untuk konteks medis
- melakukan error analysis
- menginterpretasikan model final menggunakan SHAP

Keputusan metodologis terpenting adalah:
- ICD tidak dipakai sebagai bagian model utama
- medication tidak dipakai sebagai bagian model utama
- model utama final dibuat lebih bersih agar lebih realistis untuk pasien baru dan lebih kuat dipertahankan saat sidang

Dokumen ini bisa dipakai sebagai dasar untuk menulis bab hasil, pembahasan, dan bagian metodologi final.
