import type { Metadata } from 'next';
import { Public_Sans } from 'next/font/google';
import Link from 'next/link';
import './globals.css';

const font = Public_Sans({ subsets: ['latin'], display: 'swap' });

export const metadata: Metadata = {
  title: 'AutoMLens',
  description: 'Audit AutoML results on clinical tabular data: tuning, leakage, bias and explanations.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={font.className}>
        <header className="topbar">
          <div className="shell">
            <strong>AutoMLens</strong>
            <nav style={{ display: 'flex', gap: 20 }}>
              <Link href="/">Results</Link>
              <Link href="/predict">Predict</Link>
              <Link href="/model-card">Model card</Link>
            </nav>
          </div>
        </header>
        <div className="banner"><div className="shell">Educational demo. Not medical advice and not for diagnosis.</div></div>
        <div className="shell">{children}</div>
      </body>
    </html>
  );
}
