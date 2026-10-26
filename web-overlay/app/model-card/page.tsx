import { getText } from '@/lib/api';

export default async function ModelCard() {
  const t = await getText('/model-card');
  return (
    <main>
      <h1>Model card</h1>
      {t ? <div className="panel"><pre className="card">{t}</pre></div>
         : <p className="lede">The model card is not available. Run the pipeline and start the API first.</p>}
    </main>
  );
}
