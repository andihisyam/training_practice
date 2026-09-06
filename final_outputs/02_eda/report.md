# EDA Report

- Dataset: `D:\Ratih\PracticeFusion\outputs\app\prepare_data\final_dataset.csv`
- Shape: `9916 x 2510`

## Distribusi Label

- `DMIndicator=0`: count=`8017`, pct=`80.85`
- `DMIndicator=1`: count=`1899`, pct=`19.15`

## Quality Checks

- `Jumlah pasien final`: `9916`
- `Jumlah fitur final`: `2508`

## Top Diagnosis Mean Differences

- `DiagnosisCount`: |diff|=`2.2169`
- `VisitCount`: |diff|=`1.0506`
- `Icd9_390-459`: |diff|=`0.8238`
- `Icd9_240-279`: |diff|=`0.7088`
- `Icd9_780-799`: |diff|=`0.2410`
- `Icd9_710-739`: |diff|=`0.2012`
- `Icd9_580-629`: |diff|=`0.1135`
- `Icd9_290-319`: |diff|=`0.1049`
- `Icd9_E-V`: |diff|=`0.1027`
- `Icd9_520-579`: |diff|=`0.0983`

## Top Physician Mean Differences

- `PhySp_Internal_Medicine`: |diff|=`2.9093`
- `PhySp_General_Practice`: |diff|=`0.7175`
- `PhySp_Unknown`: |diff|=`0.2334`
- `PhySp_Psychiatry`: |diff|=`0.2036`
- `PhySp_Geriatric_Medicine`: |diff|=`0.1817`
- `PhySp_Obstetrics_Gynecology`: |diff|=`0.1680`
- `PhySp_Nephrology`: |diff|=`0.1266`
- `PhySp_Endocrinology__Diabetes__Metabolism`: |diff|=`0.1259`
- `PhySp_Family_Practice`: |diff|=`0.1004`
- `PhySp_Pain_Medicine`: |diff|=`0.0964`

## Kesimpulan Singkat

- Distribusi label masih tidak seimbang, sehingga recall, F1-score, dan PR-AUC penting pada tahap modelling.
- Fitur transcript hasil rebuild tetap memberi sinyal klinis yang berguna, terutama pada BMI, berat badan, dan tekanan darah.
- Diagnosis, physician specialty, dan medication tetap perlu diawasi karena dapat mendekati label diabetes terlalu kuat.