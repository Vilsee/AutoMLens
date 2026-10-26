import SplineHero from '@/components/SplineHero';
import PredictForm from '@/components/PredictForm';
import { SCENE_B } from '@/lib/scenes';

export default function PredictPage() {
  return (
    <>
      <SplineHero scene={SCENE_B} label="Interactive 3D scene" />
      <PredictForm />
    </>
  );
}
