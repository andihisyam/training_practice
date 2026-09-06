# Methodology Checks Report

- Created at: `2026-09-06T10:35:20+07:00`
- Dataset: `D:\Ratih\PracticeFusion\outputs\app\prepare_data\final_dataset.csv`
- Overall status: `pass`
- Development/Test split: `70/30` dengan random_state `42`
- CV strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` pada development saja

## Checks

- `one_patient_one_row`: `pass` - Duplicated PatientGuid rows: 0
- `development_test_patient_disjoint`: `pass` - PatientGuid overlap count: 0
- `reproducible_split`: `pass` - Same random_state=42 produces identical index split: True
- `test_set_never_enters_cv`: `pass` - Validation/test index overlap across folds: 0
- `no_forbidden_features__clinical_core`: `pass` - No forbidden feature found.
- `no_forbidden_features__clinical_core_extreme`: `pass` - No forbidden feature found.
- `no_forbidden_features__clinical_core_extreme_weight`: `pass` - No forbidden feature found.
- `no_forbidden_features__full_transcript_comparator`: `pass` - No forbidden feature found.
- `primary_set_b_explicit_allowlist`: `pass` - Missing: []; extra: []
- `smote_train_fold_only_design`: `pass` - SMOTE comparison uses collect_cv_results with train_idx/val_idx per fold and fit_resample is applied only after selecting X_train.

## Primary Feature Sets

- `clinical_core` (Set A - Clinical Core): `5` encoded/source columns sebelum preprocessing.
- `clinical_core_extreme` (Set B - Clinical Core + Extreme): `8` encoded/source columns sebelum preprocessing.
- `clinical_core_extreme_weight` (Set C - Core + Weight): `10` encoded/source columns sebelum preprocessing.
- `full_transcript_comparator` (Set D - Full Transcript Comparator): `37` encoded/source columns sebelum preprocessing.