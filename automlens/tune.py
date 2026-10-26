"""Step 2: baseline RandomForest + Grid / Random / Optuna tuning with timing."""
import time
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
CV = StratifiedKFold(5, shuffle=True, random_state=42)


def mk(**p):
    return RF(random_state=42, n_jobs=-1, **p)


def run_all(Xtr, ytr, Xte, yte, n_iter=30, n_trials=30):
    def base():
        return mk().fit(Xtr, ytr), cross_val_score(mk(), Xtr, ytr, cv=CV, scoring='f1').mean()

    def grid():
        g = {'n_estimators': [100, 300], 'max_depth': [6, 10, None], 'min_samples_leaf': [1, 5]}
        s = GridSearchCV(mk(), g, cv=CV, scoring='f1').fit(Xtr, ytr)
        return s.best_estimator_, s.best_score_

    def rand():
        d = {'n_estimators': list(range(100, 600, 50)), 'max_depth': [4, 6, 8, 10, 14, None],
             'min_samples_leaf': list(range(1, 20)), 'max_features': ['sqrt', 'log2', 0.5]}
        s = RandomizedSearchCV(mk(), d, n_iter=n_iter, cv=CV, scoring='f1', random_state=42).fit(Xtr, ytr)
        return s.best_estimator_, s.best_score_

    def opt():
        def obj(t):
            p = dict(n_estimators=t.suggest_int('n_estimators', 100, 600),
                     max_depth=t.suggest_int('max_depth', 3, 16),
                     min_samples_leaf=t.suggest_int('min_samples_leaf', 1, 20),
                     max_features=t.suggest_categorical('max_features', ['sqrt', 'log2', 0.5]))
            return cross_val_score(mk(**p), Xtr, ytr, cv=CV, scoring='f1').mean()
        st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        st.optimize(obj, n_trials=n_trials)
        return mk(**st.best_params).fit(Xtr, ytr), st.best_value

    rows = []
    for name, fn in [('baseline', base), ('grid', grid), ('random', rand), ('optuna', opt)]:
        t = time.time()
        model, cv = fn()
        sec = round(time.time() - t, 1)
        rows.append(dict(method=name, cv_f1=float(cv), test_f1=float(f1_score(yte, model.predict(Xte))),
                         seconds=sec, model=model))
        print(f'[tune] {name}: cv_f1={cv:.4f} seconds={sec}')
    return pd.DataFrame(rows)
