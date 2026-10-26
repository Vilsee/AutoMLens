"""Step 3: FLAML and AutoGluon under the same time budget."""
import os
import shutil
import time
from sklearn.metrics import f1_score


def run_flaml(tr, te, label, budget, seed=42):
    from flaml import AutoML
    a = AutoML()
    t = time.time()
    a.fit(X_train=tr.drop(columns=label), y_train=tr[label], task='classification',
          metric='f1', time_budget=budget, seed=seed, verbose=0)
    sec = round(time.time() - t, 1)
    f1 = float(f1_score(te[label], a.predict(te.drop(columns=label))))
    info = dict(tool='FLAML', test_f1=f1, seconds=sec, winner=a.best_estimator,
                models_tried=[k for k, v in a.best_config_per_estimator.items() if v is not None],
                best_config=a.best_config,
                notes='Searches learners (LightGBM, XGBoost, RF, ExtraTrees, ...) and hyperparameters; '
                      'light preprocessing only. Check the log for what it applied in your run.')
    print(f'[flaml] winner={a.best_estimator} f1={f1:.4f} seconds={sec}')
    return a, info


def run_autogluon(tr, te, label, budget, path=None):
    from autogluon.tabular import TabularPredictor
    path = path or os.environ.get('AUTOMLENS_AG_PATH', 'artifacts/ag')
    shutil.rmtree(path, ignore_errors=True)
    t = time.time()
    p = TabularPredictor(label=label, eval_metric='f1', path=path).fit(
        tr, time_limit=budget, presets='medium_quality',
        excluded_model_types=['NN_TORCH', 'FASTAI', 'XGB', 'CAT'])
    sec = round(time.time() - t, 1)
    f1 = float(f1_score(te[label], p.predict(te.drop(columns=label))))
    lb = p.leaderboard(te, silent=True)
    info = dict(tool='AutoGluon', test_f1=f1, seconds=sec, winner=p.model_best,
                models_tried=lb['model'].tolist(),
                leaderboard=lb[['model', 'score_test', 'score_val', 'fit_time']].head(10).to_dict('records'),
                features_used=str(p.features()),
                feature_types=str(p.feature_metadata.get_type_map_raw()),
                decision_threshold=float(p.decision_threshold),
                notes='medium_quality preset (single level, no stacking) to fit in laptop memory; '
                      'automatic type-aware feature generation (4 binary columns converted to boolean, '
                      'no features created or dropped); neural nets, CatBoost and XGBoost excluded; '
                      'decision threshold calibrated for F1 on validation data.')
    print(f'[autogluon] winner={p.model_best} f1={f1:.4f} seconds={sec}')
    shutil.rmtree(path, ignore_errors=True)   # results are saved in `info`; free the disk
    return p, info
