# Ringkasan Eksperimen dan Hasil Penelitian

Dokumen ini merangkum rancangan eksperimen setelah refactor metodologi. Hasil lama tetap dipertahankan sebagai **preliminary / exploratory experiments**, sedangkan hasil final perlu dibaca dari artefak yang dihasilkan setelah pipeline baru dijalankan ulang.

## 1. Research Task

Penelitian ini adalah **binary classification** untuk mengidentifikasi status Diabetes Mellitus Tipe 2 (T2DM) berdasarkan data EHR patient-level.

Target:

```text
DMIndicator
0 = Non-DM
1 = DM
```

Konteks penggunaannya adalah screening/identifikasi status, sehingga recall atau sensitivity penting. Namun model tidak dipilih hanya dari recall karena model yang memprediksi semua pasien sebagai DM bisa menghasilkan recall tinggi tetapi tidak berguna.

Primary metric untuk model selection adalah `PR-AUC / Average Precision`.

## 2. Target Definition

Target `DMIndicator` dibaca sebagai status DM/non-DM pasien pada dataset.

Penelitian ini tidak diklaim sebagai prediksi kejadian T2DM di masa depan karena dataset tidak menyediakan tanggal diagnosis T2DM pertama yang cukup jelas untuk membuat observation window dan prediction window.

## 3. Excluded Features and Rationale

Model utama sengaja tidak memakai:
- `State`
- `PracticeGuid`
- `PatientGuid`
- `PhySp_*`
- diagnosis / ICD
- medication

Alasannya:
- `State` dapat menjadi proxy lokasi, practice, dan pola pencatatan EHR.
- `PracticeGuid` dan `PatientGuid` adalah metadata, bukan karakteristik klinis.
- `PhySp_*` dapat menjadi proxy jalur layanan kesehatan setelah pasien diketahui sakit.
- diagnosis / ICD terlalu dekat dengan outcome.
- medication dapat langsung mengindikasikan pasien DM atau proses monitoring DM.

Fitur tersebut tidak selalu salah secara statistik, tetapi tidak dipakai pada primary model agar hasil lebih defensible secara metodologi.

Leakage check refactor terbaru dengan XGBoost menunjukkan:
- clean transcript comparator PR-AUC `0.4416`
- dengan diagnosis PR-AUC `0.4820`
- dengan medication PR-AUC `0.4753`
- full comparator PR-AUC `0.5024`
- delta full vs clean `+0.0608`

Kenaikan ini mendukung keputusan bahwa diagnosis dan medication sebaiknya tidak menjadi predictor primary model.

## 4. Candidate Feature Sets

Feature set dibuat sebagai allow-list eksplisit.

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

Tujuan: kandidat utama yang tetap sederhana tetapi menangkap kondisi tipikal dan nilai tertinggi yang pernah tercatat.

### Set C - Core + Weight

Set B ditambah:

```text
Weight_Mean
Weight_Max
```

Tujuan: menguji apakah berat badan masih menambah informasi setelah BMI digunakan.

### Set D - Full Transcript Comparator

Memakai fitur transcript yang lebih lengkap, tetapi tetap tanpa State, physician specialty, diagnosis, medication, `PatientGuid`, dan `PracticeGuid`.

Tujuan: pembanding kompleks untuk melihat apakah fitur transcript tambahan memberi peningkatan yang cukup berarti.

## 5. Fixed Development/Test Split

Dataset dibagi menjadi:
- 70% development data
- 30% locked final test data

Split dibuat dengan:

```python
train_test_split(..., test_size=0.30, stratify=y, random_state=42)
```

Development data dipakai untuk semua keputusan eksperimen. Locked test hanya dipakai setelah pipeline final dibekukan.

Artefak split:
- `outputs/app/train/locked_split_manifest.json`
- `outputs/app/train/development_patient_ids.csv`
- `outputs/app/train/test_patient_ids.csv`

## 6. Feature Set Comparison

Command:

```powershell
python main.py feature-set-screen
```

Tahap ini membandingkan Set A, B, C, dan D memakai model anchor yang sama. Tujuannya supaya efek feature set tidak bercampur dengan efek algoritma.

Aturan seleksi:
- prioritas utama `mean PR-AUC`
- lihat stabilitas antar-fold
- lihat recall karena konteks screening
- jika performa set kompleks hanya naik sangat kecil, pilih set yang lebih sederhana

Output:
- `outputs/app/train/feature_set_screening_summary.csv`
- `outputs/app/train/feature_set_screening_report.md`
- `outputs/app/train/selected_feature_set.json`

Hasil run refactor terbaru:
- `full_transcript_comparator`: mean PR-AUC `0.4416 +/- 0.0048`, recall `0.6998`
- `clinical_core_extreme_weight`: mean PR-AUC `0.4400 +/- 0.0213`, recall `0.7186`
- `clinical_core_extreme`: mean PR-AUC `0.4393 +/- 0.0312`, recall `0.7194`
- `clinical_core`: mean PR-AUC `0.4221 +/- 0.0218`, recall `0.7352`

Feature set yang dipilih adalah `clinical_core_extreme` karena selisih PR-AUC terhadap set paling tinggi hanya `0.0023`, sedangkan Set B jauh lebih sederhana dan lebih mudah dijelaskan secara klinis.

## 7. Model Comparison

Command:

```powershell
python main.py cv-main
```

Setelah feature set dipilih, enam algoritma dibandingkan pada feature set yang sama:
- Logistic Regression
- SVM
- KNN
- Gradient Boosting
- XGBoost
- LightGBM

Semua memakai development CV yang sama agar perbandingan algoritma fair.

Output:
- `outputs/app/train/main_cv_fold_results.csv`
- `outputs/app/train/main_cv_summary.csv`
- `outputs/app/train/main_cv_report.md`

Hasil run refactor terbaru pada `clinical_core_extreme`:
- XGBoost: mean PR-AUC `0.4393`, recall `0.7194`, F2 `0.5886`
- Gradient Boosting: mean PR-AUC `0.4269`, recall `0.7276`, F2 `0.5915`
- LightGBM: mean PR-AUC `0.4216`, recall `0.6576`, F2 `0.5605`
- Logistic Regression: mean PR-AUC `0.4175`, recall `0.7103`, F2 `0.5882`
- SVM: mean PR-AUC `0.4153`, recall `0.7427`, F2 `0.6031`
- KNN: mean PR-AUC `0.3756`, recall `0.1798`, F2 `0.2045`

Berdasarkan primary metric `PR-AUC`, XGBoost menjadi model utama.

## 8. Imbalance Handling

Command:

```powershell
python main.py weight-compare
python main.py smote-compare
```

Eksperimen imbalance dipisahkan:
- baseline tanpa SMOTE dan tanpa class weight
- class weight
- SMOTE tanpa class weight

SMOTE diterapkan hanya pada training fold di dalam cross-validation. Validation fold tidak di-resample.

Output:
- `outputs/app/train/weighting_comparison_summary.csv`
- `outputs/app/train/weighting_comparison_report.md`
- `outputs/app/train/smote_comparison_summary.csv`
- `outputs/app/train/smote_comparison_report.md`

Hasil run refactor terbaru:
- Weighted XGBoost PR-AUC `0.4393`, recall `0.7194`, F2 `0.5886`
- Unweighted XGBoost PR-AUC `0.4366`, recall `0.1475`, F2 `0.1727`
- No SMOTE XGBoost PR-AUC `0.4366`, recall `0.1475`, F2 `0.1727`
- SMOTE XGBoost PR-AUC `0.4231`, recall `0.5711`, F2 `0.5210`

Keputusan: class weighting lebih sesuai sebagai strategi utama. SMOTE meningkatkan recall dibanding unweighted, tetapi menurunkan PR-AUC dan masih kalah dari weighted model untuk konteks screening.

## 9. Optuna

Command:

```powershell
python main.py optuna-tune
```

Optuna dijalankan hanya pada model kandidat terpilih dan development CV. Objective yang digunakan adalah `PR-AUC`.

Output:
- `outputs/app/train/optuna_trials.csv`
- `outputs/app/train/optuna_best_params.json`
- `outputs/app/train/optuna_baseline_vs_tuned.csv`
- `outputs/app/train/optuna_report.md`

Hasil run refactor terbaru:
- Baseline XGBoost PR-AUC `0.4393`
- Tuned XGBoost PR-AUC `0.4423`
- Delta PR-AUC `+0.0030`
- Tuned recall `0.7299`

Optuna memberi peningkatan kecil. Ini tetap berguna karena menunjukkan hyperparameter tuning dilakukan secara terkontrol pada development CV.

## 10. Threshold Selection

Command:

```powershell
python main.py threshold-tune
```

Threshold dicari memakai out-of-fold prediction pada development data. Final test tidak digunakan untuk memilih threshold.

Metric yang dilihat:
- recall / sensitivity
- specificity
- precision
- F1
- F2

Output:
- `outputs/app/train/threshold_tuning_results.csv`
- `outputs/app/train/threshold_tuning_best_thresholds.csv`
- `outputs/app/train/threshold_tuning_report.md`

Hasil run refactor terbaru:
- threshold terpilih `0.35`
- recall development OOF `0.8781`
- precision development OOF `0.2919`
- specificity development OOF `0.4955`
- F2 development OOF `0.6265`

Threshold `0.35` dipilih karena konteks screening lebih memprioritaskan pengurangan false negative.

## 11. Final Test Evaluation

Command:

```powershell
python main.py train
```

Setelah feature set, model, imbalance strategy, hyperparameter, dan threshold sudah ditetapkan, model dilatih pada seluruh development data dan dievaluasi sekali pada locked test.

Metric final yang dilaporkan:
- PR-AUC
- ROC-AUC
- recall / sensitivity
- specificity
- precision
- NPV
- F1
- F2
- confusion matrix
- Brier score

Output:
- `outputs/app/train/final_test_results.csv`
- `outputs/app/train/final_test_predictions.csv`
- `outputs/app/train/report.md`
- `outputs/app/train/best_model.pkl`

Hasil final locked test setelah refactor:
- model: `XGBoost`
- feature set: `clinical_core_extreme`
- threshold: `0.35`
- PR-AUC: `0.4222`
- ROC-AUC: `0.7678`
- recall: `0.8737`
- specificity: `0.4915`
- precision: `0.2894`
- NPV: `0.9426`
- F1: `0.4347`
- F2: `0.6223`
- Brier score: `0.1927`
- confusion matrix: TN `1182`, FP `1223`, FN `72`, TP `498`

## 12. Calibration

Calibration curve dan Brier score dibuat pada tahap final evaluation.

Output:
- `outputs/app/train/calibration_*.png`
- `outputs/app/train/calibration_*.csv`

Calibration membantu membaca apakah probabilitas model cukup masuk akal, bukan hanya ranking pasiennya.

## 13. SHAP

Command:

```powershell
python main.py xai
```

SHAP dijalankan pada model final setelah semua keputusan metodologi selesai.

Output:
- `outputs/app/train/xai/xai_report.md`
- `outputs/app/train/xai/xai_top_features.csv`
- `outputs/app/train/xai/shap_summary_plot.png`
- `outputs/app/train/xai/shap_summary_bar.png`
- `outputs/app/train/xai/xai_force_*.html`

Top fitur SHAP run refactor terbaru:
- `Age`
- `BMI_Max`
- `DiastolicBP_Mean`
- `BMI_Mean`
- `SystolicBP_Max`
- `Gender_F`
- `SystolicBP_Mean`
- `DiastolicBP_Max`
- `Gender_M`

Catatan interpretasi:
- SHAP menjelaskan kontribusi fitur terhadap output model.
- SHAP tidak boleh ditulis sebagai bukti sebab-akibat medis.

## 14. Limitations

Keterbatasan yang perlu ditulis:
- Tidak ada tanggal diagnosis T2DM pertama, sehingga penelitian adalah status classification, bukan future-event prediction.
- Belum ada external validation pada rumah sakit atau practice yang sepenuhnya berbeda.
- Random patient split belum membuktikan generalisasi antar-fasilitas.
- Fitur State, physician specialty, diagnosis, dan medication sengaja dikeluarkan dari primary model untuk mengurangi risiko shortcut learning.
- SHAP menjelaskan perilaku model, bukan kausalitas.

## 15. Preliminary Results

Hasil sebelum refactor metodologi tetap berguna sebagai catatan eksplorasi. Namun hasil tersebut tidak diposisikan sebagai final karena sebagian eksperimen lama masih memakai feature set yang lebih luas, termasuk physician specialty, diagnosis, atau medication.

Setelah refactor ini, hasil final yang layak dipakai untuk penulisan utama adalah hasil dari urutan:

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

Atau langsung:

```powershell
python main.py all
```
