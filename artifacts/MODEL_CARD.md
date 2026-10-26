# Model Card: cardiovascular risk classifier (AutoMLens)

## Model details
- Model: scikit-learn RandomForest, tuned with optuna
- Version: 0.1.0 | Trained: 2026-10-26 14:09 UTC
- Task: binary classification of cardiovascular disease (`cardio`)

## Data
- Source: Cardiovascular Disease dataset (Kaggle, Svetlana Ulianova), 70000 raw rows, 67936 after cleaning
- Cleaning: removed impossible blood pressure, height and weight values and duplicates; age converted to years; BMI derived
- Split: stratified 80/20 (train 54348, test 13588), seed 42
- Positive class share: 49.7%

## Metrics
- 5-fold CV F1 (train): 0.7247
- Held-out test F1: 0.7140
- CV vs test gap: +0.0107

| Gender (1=women, 2=men) | Selection rate | TPR | FPR |
|---|---|---|---|
| 1 | 0.465 | 0.688 | 0.240 |
| 2 | 0.462 | 0.694 | 0.239 |

## Evaluation notes
- Leakage canary: F1 0.971 with a leaky feature vs 0.504 once it is shuffled. AutoML exploited it: True.
- Most influential features (mean |SHAP|): ap_hi, age_years, ap_lo, cholesterol, bmi

## Intended use
Educational demonstration of an audited AutoML workflow. Not a medical device and not for diagnosis or treatment decisions.

## Known limits
- Single public dataset with self-reported features; no external validation.
- No calibration analysis; probabilities are not clinical risk scores.
- Gender is binary in the source data; other groups and intersections were not audited.
- Associations are observational, not causal.
- AutoML budgets and seeds were fixed, so rankings may change with more time or other hardware.
