export const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export async function getJson<T = any>(path: string): Promise<T | null> {
  try {
    const r = await fetch(`${API}${path}`, { cache: 'no-store' });
    return r.ok ? ((await r.json()) as T) : null;
  } catch {
    return null;
  }
}

export async function getText(path: string): Promise<string | null> {
  try {
    const r = await fetch(`${API}${path}`, { cache: 'no-store' });
    return r.ok ? await r.text() : null;
  } catch {
    return null;
  }
}
