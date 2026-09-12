import { Badge } from "@radix-ui/themes";

type Tone = "good" | "warning" | "serious" | "critical" | "neutral";

const TONES: Record<string, Tone> = {
  completed: "good",
  up: "good",
  running: "warning",
  busy: "warning",
  queued: "neutral",
  standby: "neutral",
  pending: "neutral",
  cancelled: "neutral",
  failed: "critical",
  down: "critical",
};

const COLORS: Record<Tone, "green" | "amber" | "orange" | "red" | "gray"> = {
  good: "green",
  warning: "amber",
  serious: "orange",
  critical: "red",
  neutral: "gray",
};

const ICONS: Record<Tone, string> = { good: "●", warning: "◐", serious: "▲", critical: "✕", neutral: "○" };

/** Status colour never carries meaning alone: every badge pairs an icon with the label. */
export function StatusBadge({ status }: { status: string }) {
  const tone = TONES[status] ?? "neutral";
  return (
    <Badge color={COLORS[tone]} variant="soft" radius="small" size="1">
      <span aria-hidden="true" style={{ fontSize: 9 }}>{ICONS[tone]}</span> {status}
    </Badge>
  );
}
