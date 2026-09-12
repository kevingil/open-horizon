import { Flex, Select, Switch, Text } from "@radix-ui/themes";
import { useCallback, useMemo, useState } from "react";
import { EventTimeline } from "../components/EventTimeline";
import { Panel } from "../components/Panel";
import type { EventEnvelope } from "../lib/events";
import { useEvents } from "../lib/hub";
import { useRecentEvents } from "../lib/query";

const KINDS = ["", "rollout.", "step.", "progress.", "reward.", "training.", "adapter.", "eval.", "node.", "log.", "budget."];
const MAX_LIVE = 1000;

export function EventsPage() {
  const [kind, setKind] = useState("");
  const [live, setLive] = useState(true);
  const recent = useRecentEvents(kind || undefined);
  const [tail, setTail] = useState<EventEnvelope[]>([]);
  const onEvent = useCallback((e: EventEnvelope) => {
    setTail((prev) => (prev.length > MAX_LIVE ? [...prev.slice(prev.length - MAX_LIVE / 2), e] : [...prev, e]));
  }, []);
  useEvents(onEvent);
  const events = useMemo(() => {
    const seen = new Set<number>();
    const merged = [...(recent.data ?? []), ...tail].filter((e) => (kind ? e.kind.startsWith(kind) : true)).filter((e) => (seen.has(e.seq) ? false : (seen.add(e.seq), true)));
    merged.sort((a, b) => b.seq - a.seq);
    return live ? merged : merged.filter((e) => (recent.data ?? []).some((r) => r.seq === e.seq));
  }, [recent.data, tail, kind, live]);
  return (
    <>
      <Flex align="center" justify="between" wrap="wrap" gap="2">
        <Text size="4" weight="bold">Events</Text>
        <Flex align="center" gap="3">
          <Select.Root size="1" value={kind || "all"} onValueChange={(v) => setKind(v === "all" ? "" : v)}>
            <Select.Trigger />
            <Select.Content>{KINDS.map((k) => <Select.Item key={k || "all"} value={k || "all"}>{k || "all kinds"}</Select.Item>)}</Select.Content>
          </Select.Root>
          <Text size="1" className="muted">live tail</Text>
          <Switch size="1" checked={live} onCheckedChange={setLive} />
        </Flex>
      </Flex>
      <Panel title={`${events.length} events`} right="newest first · durable log with sequence numbers">
        <EventTimeline events={events} showSubject scroll />
      </Panel>
    </>
  );
}
