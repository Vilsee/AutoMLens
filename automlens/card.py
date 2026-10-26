"""Step 5: model card generator."""


def write_card(r, path):
    best = r['export']
    g = r['bias']['by_group']
    rows = '\n'.join(f"| {k} | {v['selection_rate']:.3f} | {v['tpr']:.3f} | {v['fpr']:.3f} |" for k, v in g.items())
    top = ', '.join(s['feature'] for s in r['shap'][:5])
    c = r['canary']
    limits = '\n'.join('- ' + x for x in r['limitations'])
    m = r['meta']
    text = f"""# Model Card: cardiovascular risk classifier (AutoMLens)

## Model details
- Model: scikit-learn RandomForest, tuned with {best['method']}
- Version: {m['version']} | Trained: {m['trained_at']}
- Task: binary classification of cardiovascular disease (`cardio`)

## Data
- Source: Cardiovascular Disease dataset (Kaggle, Svetlana Ulianova), {m['rows_raw']} raw rows, {m['rows_clean']} after cleaning
- Cleaning: removed impossible blood pressure, height and weight values and duplicates; age converted to years; BMI derived
- Split: stratified 80/20 (train {m['rows_train']}, test {m['rows_test']}), seed {m['seed']}
- Positive class share: {r['balance']['positive_share']:.1%}

## Metrics
- 5-fold CV F1 (train): {best['cv_f1']:.4f}
- Held-out test F1: {best['test_f1']:.4f}
- CV vs test gap: {best['cv_f1'] - best['test_f1']:+.4f}

| Gender (1=women, 2=men) | Selection rate | TPR | FPR |
|---|---|---|---|
{rows}

## Evaluation notes
- Leakage canary: F1 {c['f1_with_leak']:.3f} with a leaky feature vs {c['f1_leak_shuffled']:.3f} once it is shuffled. AutoML exploited it: {c['exploited']}.
- Most influential features (mean |SHAP|): {top}

## Intended use
Educational demonstration of an audited AutoML workflow. Not a medical device and not for diagnosis or treatment decisions.

## Known limits
{limits}
"""
    open(path, 'w', encoding='utf-8').write(text)
