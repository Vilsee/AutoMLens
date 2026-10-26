import Spline from '@splinetool/react-spline/next';

// Full-screen fixed backdrop behind the page content (decorative).
export default function SplineHero({ scene }: { scene: string; label?: string }) {
  return (
    <div className="spline-hero" aria-hidden="true">
      <Spline scene={scene} />
    </div>
  );
}
