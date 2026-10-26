import { getText } from '@/lib/api';
import SplineHero from '@/components/SplineHero';
import { SCENE_A } from '@/lib/scenes';

export default async function ModelCard() {
  const t = await getText('/model-card');
  return (
    <>
      <SplineHero scene={SCENE_A} label="Interactive 3D scene" />
      <main>
        <h1>Model card</h1>
        {t ? <div className="panel"><pre className="card">{t}</pre></div>
           : <p className="lede">The model card is not available. Run the pipeline and start the API first.</p>}
      </main>
    </>
  );
}
