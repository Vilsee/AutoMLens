'use client';
import { useState } from 'react';
import { API } from '@/lib/api';

const FIELDS: [string, string, number, number?, number?][] = [
  ['age_years', 'Age (years)', 52, 18, 100],
  ['height', 'Height (cm)', 165, 120, 220],
  ['weight', 'Weight (kg)', 70, 30, 200],
  ['ap_hi', 'Systolic pressure (mmHg)', 130, 70, 250],
  ['ap_lo', 'Diastolic pressure (mmHg)', 85, 40, 150],
];

export default function Predict() {
  const [v, setV] = useState<Record<string, number>>({
    age_years: 52, height: 165, weight: 70, ap_hi: 130, ap_lo: 85,
    gender: 1, cholesterol: 1, gluc: 1, smoke: 0, alco: 0, active: 1,
  });
  const [out, setOut] = useState<any>(null);
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);
  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setV({ ...v, [k]: Number(e.target.value) });

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true); setErr(''); setOut(null);
    try {
      const r = await fetch(`${API}/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(v) });
      if (!r.ok) throw new Error(`The API rejected the input (${r.status}). Check the ranges.`);
      setOut(await r.json());
    } catch (x: any) {
      setErr(x.message || 'Could not reach the API. Check that it is running and NEXT_PUBLIC_API_URL is correct.');
    } finally { setBusy(false); }
  }

  const sel = (k: string, label: string, opts: [number, string][]) => (
    <label>{label}<select value={v[k]} onChange={set(k)}>{opts.map(([n, t]) => <option key={n} value={n}>{t}</option>)}</select></label>
  );
  const yn: [number, string][] = [[0, 'No'], [1, 'Yes']];
  const lvl: [number, string][] = [[1, 'Normal'], [2, 'Above normal'], [3, 'Well above normal']];
  const maxImpact = Math.max(0.0001, ...(out?.contributions ?? []).map((c: any) => Math.abs(c.impact)));

  return (
    <main>
      <h1>Estimate cardiovascular risk</h1>
      <p className="lede">Enter routine measurements. The result is a model estimate for demonstration only.</p>
      <form className="grid panel" onSubmit={submit}>
        {FIELDS.map(([k, label, , min, max]) => (
          <label key={k}>{label}<input type="number" step="any" min={min} max={max} value={v[k]} onChange={set(k)} required /></label>
        ))}
        {sel('gender', 'Sex in dataset', [[1, 'Woman'], [2, 'Man']])}
        {sel('cholesterol', 'Cholesterol', lvl)}
        {sel('gluc', 'Glucose', lvl)}
        {sel('smoke', 'Smokes', yn)}
        {sel('alco', 'Drinks alcohol', yn)}
        {sel('active', 'Physically active', yn)}
        <div style={{ alignSelf: 'end' }}><button className="primary" disabled={busy}>{busy ? 'Calculating' : 'Estimate risk'}</button></div>
      </form>
      {err && <p className="error" role="alert">{err}</p>}
      {out && (
        <section className="result panel" aria-live="polite">
          <div>Estimated probability</div>
          <div className="risk">{(out.risk * 100).toFixed(1)}%</div>
          {out.contributions.length > 0 && (
            <table><tbody>
              {out.contributions.map((c: any) => (
                <tr key={c.feature}>
                  <td>{c.feature}</td>
                  <td className="num">{c.impact > 0 ? 'raises risk' : 'lowers risk'}</td>
                  <td style={{ width: '50%' }}><div className="bar"><span className={c.impact > 0 ? 'bad' : ''} style={{ width: `${(Math.abs(c.impact) / maxImpact) * 100}%` }} /></div></td>
                </tr>
              ))}
            </tbody></table>
          )}
        </section>
      )}
    </main>
  );
}
