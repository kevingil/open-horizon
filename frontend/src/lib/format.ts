export function fmtNumber(n: number | null | undefined, digits = 0): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "–";
  return n.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

export function fmtCompact(n: number | null | undefined): string {
  if (n === null || n === undefined) return "–";
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (Math.abs(n) >= 10_000) return `${(n / 1000).toFixed(0)}k`;
  if (Math.abs(n) >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return fmtNumber(n);
}

export function fmtUsd(n: number | null | undefined, digits = 4): string {
  if (n === null || n === undefined) return "–";
  return `$${n.toFixed(digits)}`;
}

export function fmtReward(n: number | null | undefined): string {
  if (n === null || n === undefined) return "–";
  return (n >= 0 ? "+" : "") + n.toFixed(3);
}

export function fmtPct(n: number | null | undefined): string {
  if (n === null || n === undefined) return "–";
  return `${(n * 100).toFixed(0)}%`;
}

export function fmtMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "–";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  return `${(ms / 60_000).toFixed(1)} min`;
}

export function fmtDuration(startIso: string | null | undefined, endIso?: string | null): string {
  if (!startIso) return "–";
  const start = Date.parse(startIso);
  const end = endIso ? Date.parse(endIso) : Date.now();
  return fmtMs(end - start);
}

export function fmtAgo(iso: string | null | undefined): string {
  if (!iso) return "–";
  const s = Math.max(0, (Date.now() - Date.parse(iso)) / 1000);
  if (s < 10) return "just now";
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86_400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86_400)}d ago`;
}

export function fmtTime(iso: string | null | undefined): string {
  if (!iso) return "–";
  return new Date(iso).toLocaleTimeString([], { hour12: false });
}

export function fmtDateTime(iso: string | null | undefined): string {
  if (!iso) return "–";
  return new Date(iso).toLocaleString([], { hour12: false });
}

/** Parse a `{tool, input}` action or `{tool, ...}` observation payload. */
export function parseJson(text: string): Record<string, unknown> | null {
  try {
    const v = JSON.parse(text);
    return typeof v === "object" && v !== null ? (v as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}
