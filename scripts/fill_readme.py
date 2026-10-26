"""Fill the results section of README.md from artifacts/results.json.

Usage (from the project root):  python scripts/fill_readme.py
Only the text between <!-- RESULTS:START --> and <!-- RESULTS:END --> is replaced.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
r = json.loads((ROOT / 'artifacts' / 'results.json').read_text(encoding='utf-8'))


def fx(x, n=4):
    return f'{x:.{n}f}'


m = r['meta']
out = [f"_Generated from `artifacts/results.json` (trained {m['trained_at']}; {m['rows_clean']:,} clean rows, "
       f"{m['rows_train']:,} train / {m['rows_test']:,} test, seed {m['seed']}, AutoML budget {m['budget_seconds']} s)._", '']

out += ['#### Manual tuning (RandomForest, 5-fold stratified CV on the training split)', '',
        '| Method | CV F1 | Test F1 | Seconds |', '|---|---:|---:|---:|']
out += [f"| {t['method']} | {fx(t['cv_f1'])} | {fx(t['test_f1'])} | {t['seconds']} |" for t in r['tuning']]

out += ['', '#### Manual vs AutoML (same split, same time cap)', '',
        '| Tool | Test F1 | Seconds | Interpretability |', '|---|---:|---:|---|']
out += [f"| {c['tool']} | {fx(c['test_f1'])} | {c['seconds']} | {c['interpretability']} |" for c in r['comparison']]

out += ['', '#### What each AutoML tool tried', '']
for t in r['automl']:
    out.append(f"- **{t['tool']}**: winner `{t['winner']}`; models tried: {', '.join(str(x) for x in t['models_tried'])}.")
    if t.get('notes'):
        out.append(f"  - Notes: {t['notes']}")
    if t.get('decision_threshold') is not None:
        out.append(f"  - Calibrated decision threshold: {t['decision_threshold']:.3f}")
    if t.get('leaderboard'):
        out += ['', f"  | {t['tool']} model | Test F1 | Validation F1 | Fit seconds |", '  |---|---:|---:|---:|']
        out += [f"  | {x['model']} | {fx(x['score_test'])} | {fx(x['score_val'])} | {fx(x['fit_time'], 1)} |"
                for x in t['leaderboard']]
        out.append('')

o, c = r['overfitting'], r['canary']
out += ['', '#### Overfitting check (exported model)', '', '| 5-fold CV F1 | Held-out test F1 | Gap |', '|---:|---:|---:|',
        f"| {fx(o['cv_f1'])} | {fx(o['test_f1'])} | {o['gap']:+.4f} |", '',
        '#### Leakage canary (a fake feature that copies the diagnosis, 3% noise)', '',
        '| Measure | Value |', '|---|---:|',
        f"| F1 with the leaky feature present | {fx(c['f1_with_leak'])} |",
        f"| F1 after shuffling the leaky feature | {fx(c['f1_leak_shuffled'])} |",
        f"| AutoML exploited the leak | {c['exploited']} |"]
if 'canary_rank' in c:
    out += [f"| Rank of the leaky feature in importance | {c['canary_rank']} |",
            f"| Share of total importance | {c['canary_importance_share']:.1%} |"]

b = r['bias']
out += ['', f"#### Fairness by gender (1 = woman, 2 = man). Positive class share: {r['balance']['positive_share']:.1%}", '',
        '| Group | Selection rate | True positive rate | False positive rate |', '|---|---:|---:|---:|']
out += [f"| {g} | {v['selection_rate']:.3f} | {v['tpr']:.3f} | {v['fpr']:.3f} |" for g, v in b['by_group'].items()]
out.append(f"| Largest gap | {b['max_gap']['selection_rate']:.3f} | {b['max_gap']['tpr']:.3f} | {b['max_gap']['fpr']:.3f} |")

out += ['', '#### Most influential features (mean absolute SHAP value)', '', '| Feature | Mean abs SHAP |', '|---|---:|']
out += [f"| {s['feature']} | {s['importance']:.4f} |" for s in r['shap']]

new = '<!-- RESULTS:START -->\n' + '\n'.join(out) + '\n<!-- RESULTS:END -->'
p = ROOT / 'README.md'
s = p.read_text(encoding='utf-8')
if '<!-- RESULTS:START -->' not in s:
    raise SystemExit('Markers not found in README.md')
p.write_text(re.sub(r'<!-- RESULTS:START -->.*?<!-- RESULTS:END -->', lambda _: new, s, flags=re.S), encoding='utf-8')
print('README.md results section updated.')
