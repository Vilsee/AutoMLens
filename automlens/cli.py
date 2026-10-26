"""Run the whole pipeline: python -m automlens.cli --budget 900"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from . import audit, automl, card, data, tune

ART = Path('artifacts')
LIMITS = ['Single public dataset with self-reported features; no external validation.',
          'No calibration analysis; probabilities are not clinical risk scores.',
          'Gender is binary in the source data; other groups and intersections were not audited.',
          'Associations are observational, not causal.',
          'AutoML budgets and seeds were fixed, so rankings may change with more time or other hardware.']


def main():
    ap = argparse.ArgumentParser(prog='automlens')
    ap.add_argument('--data', default='data/cardio_train.csv')
    ap.add_argument('--budget', type=int, default=900, help='seconds per AutoML tool')
    ap.add_argument('--canary-budget', type=int, default=120)
    ap.add_argument('--skip-autogluon', action='store_true')
    args = ap.parse_args()
    ART.mkdir(exist_ok=True)
    L = data.LABEL

    # Step 1
    df = data.load(args.data)
    tr, te = data.split(df)
    Xtr, ytr, Xte, yte = tr[data.FEATURES], tr[L], te[data.FEATURES], te[L]
    print(f'rows raw={df.attrs["raw_rows"]} clean={len(df)} train={len(tr)} test={len(te)}')

    # Step 2
    t = tune.run_all(Xtr, ytr, Xte, yte)
    tuning = t.drop(columns='model').to_dict('records')
    best_row = t.loc[t.cv_f1.idxmax()]          # chosen by CV, never by the test set
    best_model = best_row['model']
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].bar(t.method, t.test_f1); ax[0].set_title('Test F1'); ax[0].set_ylim(t.test_f1.min() - 0.02, t.test_f1.max() + 0.01)
    ax[1].bar(t.method, t.seconds); ax[1].set_title('Seconds')
    fig.tight_layout(); fig.savefig(ART / 'tuning_comparison.png', dpi=150); plt.close(fig)

    # Step 3
    flaml_model, flaml_info = automl.run_flaml(tr, te, L, args.budget)
    tools = [flaml_info]
    if not args.skip_autogluon:
        try:
            _, ag_info = automl.run_autogluon(tr, te, L, args.budget)
            tools.append(ag_info)
        except ImportError:
            print('AutoGluon not installed; skipping. pip install autogluon.tabular')
    comparison = [dict(tool=f'Manual ({best_row["method"]})', test_f1=float(best_row['test_f1']),
                       seconds=float(best_row['seconds']), interpretability='High: single forest, exact SHAP')]
    comparison += [dict(tool=x['tool'], test_f1=x['test_f1'], seconds=x['seconds'],
                        interpretability='Medium: tree ensemble, SHAP possible' if x['tool'] == 'FLAML'
                        else 'Low: stacked ensemble, hard to explain') for x in tools]

    # Step 4
    leaky_tr, leaky_te = audit.add_canary(tr, L), audit.add_canary(te, L, seed=7)
    leaky_a, _ = automl.run_flaml(leaky_tr, leaky_te, L, args.canary_budget)
    canary = audit.canary_report(leaky_a, leaky_te.drop(columns=L), leaky_te[L])
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

