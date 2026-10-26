"""Run the whole pipeline (resumable): python -m automlens.cli --budget 900"""
import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import shap

from . import audit, automl, card, data, tune

ART = Path('artifacts')
LIMITS = ['Single public dataset with self-reported features; no external validation.',
          'No calibration analysis; probabilities are not clinical risk scores.',
          'Gender is binary in the source data; other groups and intersections were not audited.',
          'Associations are observational, not causal.',
          'AutoML budgets and seeds were fixed, so rankings may change with more time or other hardware.']


def cached(path, fn, fresh):
    """Run fn once and save the result, so a later crash never loses finished work."""
    if path.exists() and not fresh:
        print(f'[resume] using saved {path.name} (pass --fresh to redo)')
        return joblib.load(path)
    out = fn()
    if out is not None:
        joblib.dump(out, path, compress=3)
    return out


def main():
    ap = argparse.ArgumentParser(prog='automlens')
    ap.add_argument('--data', default='data/cardio_train.csv')
    ap.add_argument('--budget', type=int, default=900, help='seconds per AutoML tool')
    ap.add_argument('--canary-budget', type=int, default=120)
    ap.add_argument('--skip-autogluon', action='store_true')
    ap.add_argument('--fresh', action='store_true', help='ignore saved checkpoints')
    args = ap.parse_args()
    ART.mkdir(exist_ok=True)
    L = data.LABEL

    # Step 1
    df = data.load(args.data)
    tr, te = data.split(df)
    Xtr, ytr, Xte, yte = tr[data.FEATURES], tr[L], te[data.FEATURES], te[L]
    print(f'rows raw={df.attrs["raw_rows"]} clean={len(df)} train={len(tr)} test={len(te)}')

    # Step 2 (saved after it finishes; only the best model is kept in memory)
    def do_tune():
        t = tune.run_all(Xtr, ytr, Xte, yte)
        best = t.loc[t.cv_f1.idxmax()]            # chosen by CV, never by the test set
        return dict(tuning=t.drop(columns='model').to_dict('records'),
                    best=best.drop('model').to_dict(), model=best['model'])
    ck = cached(ART / 'ckpt_tune.joblib', do_tune, args.fresh)
    tuning, best_row, best_model = ck['tuning'], ck['best'], ck['model']
    tdf = pd.DataFrame(tuning)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].bar(tdf.method, tdf.test_f1); ax[0].set_title('Test F1')
    ax[0].set_ylim(tdf.test_f1.min() - 0.02, tdf.test_f1.max() + 0.01)
    ax[1].bar(tdf.method, tdf.seconds); ax[1].set_title('Seconds')
    fig.tight_layout(); fig.savefig(ART / 'tuning_comparison.png', dpi=150); plt.close(fig)
    gc.collect()

    # Step 3
    def do_flaml():
        a, info = automl.run_flaml(tr, te, L, args.budget)
        del a
        return info
    flaml_info = cached(ART / f'ckpt_flaml_{args.budget}.joblib', do_flaml, args.fresh)
    tools = [flaml_info]
    if not args.skip_autogluon:
        def do_ag():
            gc.collect()
            try:
                return automl.run_autogluon(tr, te, L, args.budget)[1]
            except Exception as e:
                print(f'[autogluon] FAILED: {type(e).__name__}: {e}')
                return None
        ag_info = cached(ART / f'ckpt_ag_{args.budget}.joblib', do_ag, args.fresh)
        if ag_info:
            tools.append(ag_info)
        else:
            print('Continuing without AutoGluon. Fix the cause and rerun; saved steps are reused.')
    comparison = [dict(tool=f'Manual ({best_row["method"]})', test_f1=float(best_row['test_f1']),
                       seconds=float(best_row['seconds']), interpretability='High: single forest, exact SHAP')]
    comparison += [dict(tool=x['tool'], test_f1=x['test_f1'], seconds=x['seconds'],
                        interpretability='Medium: tree ensemble, SHAP possible' if x['tool'] == 'FLAML'
                        else 'Low: ensemble of several models, hard to explain') for x in tools]

    # Step 4
    gc.collect()
    leaky_tr, leaky_te = audit.add_canary(tr, L), audit.add_canary(te, L, seed=7)
    leaky_a, _ = automl.run_flaml(leaky_tr, leaky_te, L, args.canary_budget)
    canary = audit.canary_report(leaky_a, leaky_te.drop(columns=L), leaky_te[L])
    del leaky_a
    pred = best_model.predict(Xte)
    by_group, gap = audit.bias_table(yte, pred, Xte['gender'])
    shap_rank, S, sv = audit.shap_global(best_model, Xte)
    shap.summary_plot(sv, S, show=False)
    plt.tight_layout(); plt.savefig(ART / 'shap_summary.png', dpi=150); plt.close()

    # Step 5
    joblib.dump(best_model, ART / 'model.joblib', compress=3)
    results = dict(
        meta=dict(version='0.1.0', trained_at=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
                  rows_raw=df.attrs['raw_rows'], rows_clean=len(df), rows_train=len(tr), rows_test=len(te),
                  seed=data.SEED, budget_seconds=args.budget),
        tuning=tuning, automl=tools, comparison=comparison,
        overfitting=dict(cv_f1=float(best_row['cv_f1']), test_f1=float(best_row['test_f1']),
                         gap=float(best_row['cv_f1'] - best_row['test_f1'])),
        canary=canary,
        bias=dict(by_group=by_group, max_gap=gap),
        balance=dict(positive_share=float(df[L].mean())),
        shap=shap_rank, limitations=LIMITS,
        export=dict(method=best_row['method'], cv_f1=float(best_row['cv_f1']), test_f1=float(best_row['test_f1'])))
    (ART / 'results.json').write_text(json.dumps(results, indent=2, default=str), encoding='utf-8')
    card.write_card(results, ART / 'MODEL_CARD.md')
    print('Done. Artifacts written to artifacts/')


if __name__ == '__main__':
    main()
