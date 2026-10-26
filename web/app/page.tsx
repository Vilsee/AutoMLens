import { getJson } from '@/lib/api';
import SplineHero from '@/components/SplineHero';
import { SCENE_A } from '@/lib/scenes';

const pct = (n: number) => (n * 100).toFixed(1) + '%';

function Bar({ v, max = 1, bad = false }: { v: number; max?: number; bad?: boolean }) {
  return (
    <div className="bar" role="img" aria-label={`${(v / max * 100).toFixed(0)} percent`}>
      <span className={bad ? 'bad' : ''} style={{ width: `${Math.min(100, (v / max) * 100)}%` }} />
    </div>
  );
}

export default async function Home() {
  const r = await getJson('/leaderboard');
  if (!r) {
    return (
      <main>
        <h1>No results to show yet</h1>
        <p className="lede">
          The API did not respond. Run the pipeline, then start it with <code>uvicorn api.main:app --port 8000</code>,
          or set <code>NEXT_PUBLIC_API_URL</code> to your deployed API.
        </p>
      </main>
    );
  }
  const c = r.canary;
  const maxShap = r.shap[0]?.importance ?? 1;
  return (
    <main>
      <h1>Can you trust the AutoML winner?</h1>
      <p className="lede">
        {r.meta.rows_clean.toLocaleString()} patient records, one fixed split, and an identical{' '}
        {Math.round(r.meta.budget_seconds / 60)}-minute budget for every AutoML tool.
      </p>
      <SplineHero scene={SCENE_A} label="Interactive 3D scene" />

      <h2>A leaky feature made the model look near perfect</h2>
      <p className="sub">
        We added a fake feature that copies the diagnosis. {c.exploited ? 'The AutoML tool used it.' : 'The AutoML tool mostly ignored it.'}
        {' '}Once the feature is shuffled, as it would be in real use, the score falls back.
      </p>
      <div className="panel leak">
        <div>
          <div>F1 with the leaky feature</div>
          <div className="big bad-num">{pct(c.f1_with_leak)}</div>
          <Bar v={c.f1_with_leak} bad />
        </div>
        <div>
          <div>F1 after shuffling it</div>
          <div className="big ok-num">{pct(c.f1_leak_shuffled)}</div>
          <Bar v={c.f1_leak_shuffled} />
        </div>
      </div>

      <h2>Manual tuning methods</h2>
      <p className="sub">Same RandomForest, three search strategies. Cross-validated F1 and held-out test F1.</p>
      <div className="panel">
        <table>
          <thead><tr><th>Method</th><th className="num">CV F1</th><th className="num">Test F1</th><th className="num">Seconds</th><th>Test F1</th></tr></thead>
          <tbody>
            {r.tuning.map((t: any) => (
              <tr key={t.method}>
                <td>{t.method}</td><td className="num">{t.cv_f1.toFixed(4)}</td><td className="num">{t.test_f1.toFixed(4)}</td>
                <td className="num">{t.seconds}</td><td><Bar v={t.test_f1} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Manual vs AutoML</h2>
      <div className="panel">
        <table>
          <thead><tr><th>Tool</th><th className="num">Test F1</th><th className="num">Seconds</th><th>Interpretability</th></tr></thead>
          <tbody>
            {r.comparison.map((t: any) => (
              <tr key={t.tool}><td>{t.tool}</td><td className="num">{t.test_f1.toFixed(4)}</td><td className="num">{t.seconds}</td><td>{t.interpretability}</td></tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Fairness by gender</h2>
      <p className="sub">
        Positive class share is {pct(r.balance.positive_share)}. Gaps are the largest difference between groups (1 = women, 2 = men).
      </p>
      <div className="panel">
        <table>
          <thead><tr><th>Group</th><th className="num">Selection rate</th><th className="num">True positive rate</th><th className="num">False positive rate</th></tr></thead>
          <tbody>
            {Object.entries(r.bias.by_group).map(([g, m]: any) => (
              <tr key={g}><td>{g}</td><td className="num">{m.selection_rate.toFixed(3)}</td><td className="num">{m.tpr.toFixed(3)}</td><td className="num">{m.fpr.toFixed(3)}</td></tr>
            ))}
            <tr><td>Largest gap</td><td className="num">{r.bias.max_gap.selection_rate.toFixed(3)}</td><td className="num">{r.bias.max_gap.tpr.toFixed(3)}</td><td className="num">{r.bias.max_gap.fpr.toFixed(3)}</td></tr>
          </tbody>
        </table>
      </div>

      <h2>What drives the predictions</h2>
      <div className="panel">
        <table>
          <tbody>
            {r.shap.map((s: any) => (
              <tr key={s.feature}><td>{s.feature}</td><td style={{ width: '60%' }}><Bar v={s.importance} max={maxShap} /></td></tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Limitations</h2>
      <ul className="lede">{r.limitations.map((l: string) => <li key={l}>{l}</li>)}</ul>
    </main>
  );
}

