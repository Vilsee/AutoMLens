"""Step 4: leakage canary, overfitting, bias, class balance and SHAP."""
import numpy as np
import shap
from fairlearn.metrics import MetricFrame, false_positive_rate, selection_rate, true_positive_rate
from sklearn.metrics import f1_score

CANARY = 'lab_flag'


def add_canary(df, label='cardio', noise=0.03, seed=42):
    """Add a fake post-diagnosis feature that copies the label (with 3% noise)."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    d[CANARY] = (d[label] ^ (rng.random(len(d)) < noise)).astype(int)
    return d


def canary_report(a, Xte, yte):
    shuffled = Xte.copy()
    shuffled[CANARY] = np.random.default_rng(0).permutation(shuffled[CANARY].values)
    out = dict(f1_with_leak=float(f1_score(yte, a.predict(Xte))),
               f1_leak_shuffled=float(f1_score(yte, a.predict(shuffled))))
    try:
        names, imp = list(a.feature_names_in_), np.asarray(a.feature_importances_, dtype=float)
        order = list(np.argsort(-imp))
        out['canary_importance_share'] = float(imp[names.index(CANARY)] / imp.sum())
        out['canary_rank'] = int(order.index(names.index(CANARY)) + 1)
    except Exception as e:  # importance not available for this learner
        out['importance_note'] = f'unavailable: {e}'
    out['exploited'] = bool(out['f1_with_leak'] - out['f1_leak_shuffled'] > 0.10)
    return out


def bias_table(y, pred, sensitive):
    mf = MetricFrame(metrics={'selection_rate': selection_rate, 'tpr': true_positive_rate,
                              'fpr': false_positive_rate},
                     y_true=y, y_pred=pred, sensitive_features=sensitive)
    by = {str(k): {m: float(v) for m, v in row.items()} for k, row in mf.by_group.to_dict('index').items()}
    return by, {m: float(v) for m, v in mf.difference().items()}


def shap_positive(model, X):
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):
        return sv[1]
    return sv[..., 1] if sv.ndim == 3 else sv


def shap_global(model, X, n=2000):
    S = X.sample(min(n, len(X)), random_state=42)
    sv = shap_positive(model, S)
    imp = np.abs(sv).mean(axis=0)
    ranked = sorted(zip(S.columns, imp), key=lambda x: -x[1])
    return [dict(feature=f, importance=float(v)) for f, v in ranked], S, sv
