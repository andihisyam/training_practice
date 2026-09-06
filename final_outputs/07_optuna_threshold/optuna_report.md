# Optuna Tuning Report

- Created at: `2026-09-06T08:56:05+07:00`
- Dataset: `D:\Ratih\PracticeFusion\outputs\app\prepare_data\final_dataset.csv`
- Development/Test split: `70/30` dengan random_state `42`
- Development CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Feature set: `clinical_core_extreme`
- Model: `XGBoost`
- Trials: `30`
- Weighted model: `True`

## Tujuan

- Mengoptimalkan hyperparameter pada development set saja.
- Metric optimasi utama adalah `PR-AUC`, sedangkan recall dan F2 tetap dicatat untuk konteks screening.

## Hasil Terbaik

- Mean PR-AUC terbaik: `0.4423`
- Mean Recall terbaik: `0.7299`
- Mean F2 terbaik: `0.5963`
- Best params: `{'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.023623212677489378, 'subsample': 0.6040569883749647, 'colsample_bytree': 0.6750130525733469, 'min_child_weight': 8, 'reg_lambda': 9.941450502972573, 'reg_alpha': 0.1975141916850305}`

## Baseline vs Tuned

- Baseline PR-AUC: `0.4393`; Tuned PR-AUC: `0.4423`; Delta: `+0.0030`
- Baseline Recall: `0.7194`; Tuned Recall: `0.7299`

## Artifacts

- Split manifest: `D:\Ratih\PracticeFusion\outputs\app\train\locked_split_manifest.json`
- Trials CSV: `D:\Ratih\PracticeFusion\outputs\app\train\optuna_trials.csv`
- Best params JSON: `D:\Ratih\PracticeFusion\outputs\app\train\optuna_best_params.json`
- Baseline vs tuned CSV: `D:\Ratih\PracticeFusion\outputs\app\train\optuna_baseline_vs_tuned.csv`