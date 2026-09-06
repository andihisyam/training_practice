# Prepare Data Report

- Final dataset path: `D:\Ratih\PracticeFusion\outputs\app\prepare_data\final_dataset.csv`
- Transcript rebuild path: `D:\Ratih\PracticeFusion\outputs\app\prepare_data\transcript_rebuilt.csv`
- Final shape: `9916 x 2510`

## Cohort Notes
- Patient and label backbone comes from d5.csv.
- Age is retained from patient.csv because d5.csv does not provide Age directly.
- Transcript is rebuilt entirely from d2.csv and becomes the only required clinical table besides the patient backbone.
- Diagnosis, physician specialty, and medication are retained as optional left-joined aggregated blocks so the final cohort does not collapse when one block is absent.
- Optional aggregated blocks are filled with zero when a patient has no matching row in that source table.
- Final cohort uses the inner intersection of patient backbone and rebuilt transcript only.
- Transcript std columns are removed from the final dataset.
- Columns with missing rate above 40% are dropped.
- Explicit missing indicator columns are added to the final dataset for any feature that will be imputed later.

## Join Summary

- `transcript_rebuilt` (required_inner): rows=`9916`, patients=`9916`, missing_from_backbone=`32`, coverage_in_final_cohort=`0.9968`
- `diagnosis` (optional_left): rows=`9948`, patients=`9948`, missing_from_backbone=`0`, coverage_in_final_cohort=`1.0000`
- `physician_specialty` (optional_left): rows=`9948`, patients=`9948`, missing_from_backbone=`0`, coverage_in_final_cohort=`1.0000`
- `medication` (optional_left): rows=`9836`, patients=`9836`, missing_from_backbone=`112`, coverage_in_final_cohort=`0.9887`

## Dropped Patients

- `No rebuilt transcript record`: `32`
- `Dropped by required backbone + transcript intersection`: `32`

## Transcript Cleaning Summary

- `Height`: raw_missing=`0`, raw_zero=`0`, clean_missing=`10`, invalidated=`10`
- `Weight`: raw_missing=`0`, raw_zero=`0`, clean_missing=`0`, invalidated=`0`
- `BMI`: raw_missing=`0`, raw_zero=`0`, clean_missing=`2`, invalidated=`2`
- `SystolicBP`: raw_missing=`0`, raw_zero=`1230`, clean_missing=`1241`, invalidated=`1241`
- `DiastolicBP`: raw_missing=`0`, raw_zero=`1230`, clean_missing=`1268`, invalidated=`1268`
- `RespiratoryRate`: raw_missing=`20348`, raw_zero=`0`, clean_missing=`20361`, invalidated=`13`
- `Temperature`: raw_missing=`21753`, raw_zero=`0`, clean_missing=`21914`, invalidated=`161`

## Final Missing Strategy

- Missing drop threshold: `0.40`
- Dropped std columns: `0`
- Dropped high-missing columns: `0`
- Missing indicator columns added: `16`
- `SystolicBP_Min` -> `IsImputed_SystolicBP_Min`: missing_count=`22`, missing_rate=`0.0022`
- `SystolicBP_Max` -> `IsImputed_SystolicBP_Max`: missing_count=`22`, missing_rate=`0.0022`
- `SystolicBP_Mean` -> `IsImputed_SystolicBP_Mean`: missing_count=`22`, missing_rate=`0.0022`
- `SystolicBP_Change` -> `IsImputed_SystolicBP_Change`: missing_count=`22`, missing_rate=`0.0022`
- `DiastolicBP_Min` -> `IsImputed_DiastolicBP_Min`: missing_count=`22`, missing_rate=`0.0022`
- `DiastolicBP_Max` -> `IsImputed_DiastolicBP_Max`: missing_count=`22`, missing_rate=`0.0022`
- `DiastolicBP_Mean` -> `IsImputed_DiastolicBP_Mean`: missing_count=`22`, missing_rate=`0.0022`
- `DiastolicBP_Change` -> `IsImputed_DiastolicBP_Change`: missing_count=`22`, missing_rate=`0.0022`
- `RespiratoryRate_Min` -> `IsImputed_RespiratoryRate_Min`: missing_count=`3236`, missing_rate=`0.3263`
- `RespiratoryRate_Max` -> `IsImputed_RespiratoryRate_Max`: missing_count=`3236`, missing_rate=`0.3263`
- `RespiratoryRate_Mean` -> `IsImputed_RespiratoryRate_Mean`: missing_count=`3236`, missing_rate=`0.3263`
- `RespiratoryRate_Change` -> `IsImputed_RespiratoryRate_Change`: missing_count=`3236`, missing_rate=`0.3263`
- `Temperature_Min` -> `IsImputed_Temperature_Min`: missing_count=`3044`, missing_rate=`0.3070`
- `Temperature_Max` -> `IsImputed_Temperature_Max`: missing_count=`3044`, missing_rate=`0.3070`
- `Temperature_Mean` -> `IsImputed_Temperature_Mean`: missing_count=`3044`, missing_rate=`0.3070`
- `Temperature_Change` -> `IsImputed_Temperature_Change`: missing_count=`3044`, missing_rate=`0.3070`