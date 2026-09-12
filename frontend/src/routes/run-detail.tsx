import { Button, Flex, Select, Tabs, Text } from "@radix-ui/themes";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "@tanstack/react-router";
import { useCallback, useState } from "react";
import { Conversation } from "../components/Conversation";
import { EventTimeline } from "../components/EventTimeline";
import { Panel } from "../components/Panel";
import { StatTile } from "../components/StatTile";
import { StatusBadge } from "../components/StatusBadge";
import { cancelRun, fetchRubrics, rescoreRun } from "../lib/api";
import type { EventEnvelope } from "../lib/events";
import { fmtCompact, fmtDateTime, fmtDuration, fmtMs, fmtReward, fmtUsd } from "../lib/format";
import { useEvents } from "../lib/hub";
import { useRun, useRunEvents } from "../lib/query";
import type { RunDetail } from "../lib/types";

export function RunDetailPage() {
  const { runId } = useParams({ from: "/rollouts/$runId" });
  const qc = useQueryClient();
  const run = useRun(runId);
  const events = useRunEvents(runId);
  const rubrics = useQuery({ queryKey: ["rubrics"], queryFn: fetchRubrics, staleTime: Infinity });
  const [rubric, setRubric] = useState("");
  const [delta, setDelta] = useState<number | null>(null);

  // Append live steps without waiting for the refetch.
  const onEvent = useCallback(
    (e: EventEnvelope) => {
      if (e.subject?.kind !== "run" || e.subject.id !== runId || e.kind !== "step.recorded") return;
      const step = e.payload.step;
      qc.setQueryData<RunDetail>(["run", runId], (prev) => {
        if (!prev || prev.trajectory.steps.some((s) => s.index === step.index)) return prev;
        const steps = [...prev.trajectory.steps, step].sort((a, b) => a.index - b.index);
        return { ...prev, trajectory: { ...prev.trajectory, steps }, manifest: { ...prev.manifest, step_count: steps.length } };
      });
    },
    [qc, runId],
  );
  useEvents(onEvent);

  const cancel = useMutation({ mutationFn: () => cancelRun(runId) });
  const rescore = useMutation({
    mutationFn: () => rescoreRun(runId, rubric || rubrics.data?.[0]?.name || "heuristic-v1"),
    onSuccess: (r) => {
      setDelta(r.delta ?? null);
      void qc.invalidateQueries({ queryKey: ["run", runId] });
    },
  });

  if (run.error) return <div className="empty">{String(run.error)}</div>;
  if (!run.data) return <div className="empty">loading…</div>;
  const { manifest: m, task, trajectory, reward } = run.data;
  const active = m.status === "running" || m.status === "queued";
  const policySteps = trajectory.steps.filter((s) => s.actor === "policy" && s.duration_ms != null);
  const meanPolicy = policySteps.length ? policySteps.reduce((a, s) => a + (s.duration_ms ?? 0), 0) / policySteps.length : null;

  return (
    <>
      <Flex align="center" gap="3" wrap="wrap">
        <Link to="/rollouts" className="muted" style={{ fontSize: 12 }}>← rollouts</Link>
        <Text size="4" weight="bold" className="mono">{m.id}</Text>
        <StatusBadge status={m.status} />
        <Text size="1" className="muted">{m.model_id} · profile {m.profile} · {m.infra_target}</Text>
        <div style={{ marginLeft: "auto" }}>
          {active ? (
            <Button size="1" color="red" variant="soft" onClick={() => cancel.mutate()} disabled={cancel.isPending} data-testid="cancel-run">
              {cancel.isPending ? "Cancelling…" : "Cancel rollout"}
            </Button>
          ) : null}
        </div>
      </Flex>
      <Text size="2">{task.prompt}</Text>

      <div className="tiles">
        <StatTile label="Steps" value={`${m.step_count}`} sub={`horizon ${m.horizon}`} />
        <StatTile label="Tokens" value={fmtCompact(m.tokens)} sub={fmtUsd(m.cost_usd)} />
        <StatTile label="Reward" value={fmtReward(m.terminal_reward)} sub={reward ? reward.rubric : "unscored"} />
        <StatTile label="Duration" value={fmtDuration(m.started_at, m.finished_at)} sub={m.started_at ? fmtDateTime(m.started_at) : "not started"} />
        <StatTile label="Mean policy latency" value={fmtMs(meanPolicy)} sub={`${policySteps.length} turns`} />
      </div>
      {m.error ? <Text size="2" color="red">error: {m.error}</Text> : null}

      <Tabs.Root defaultValue="conversation">
        <Tabs.List>
          <Tabs.Trigger value="conversation">Conversation</Tabs.Trigger>
          <Tabs.Trigger value="reward">Reward</Tabs.Trigger>
          <Tabs.Trigger value="events">Events ({events.data?.length ?? 0})</Tabs.Trigger>
          <Tabs.Trigger value="task">Task</Tabs.Trigger>
        </Tabs.List>
        <div style={{ paddingTop: 12 }}>
          <Tabs.Content value="conversation">
            <Panel title="Trajectory" right={`${trajectory.steps.length} steps`}>
              {trajectory.errors.length ? <Text size="1" color="red">{trajectory.errors.join(" · ")}</Text> : null}
              <Conversation steps={trajectory.steps} />
            </Panel>
          </Tabs.Content>
          <Tabs.Content value="reward">
            <Panel
              title={reward ? `${reward.rubric} · ${reward.source}` : "Unscored"}
              right={
                !active && rubrics.data ? (
                  <Flex gap="2" align="center">
                    <Select.Root size="1" value={rubric || rubrics.data[0]?.name} onValueChange={setRubric}>
                      <Select.Trigger />
                      <Select.Content>{rubrics.data.map((r) => <Select.Item key={r.name} value={r.name}>{r.name}</Select.Item>)}</Select.Content>
                    </Select.Root>
                    <Button size="1" variant="soft" onClick={() => rescore.mutate()} disabled={rescore.isPending} data-testid="rescore">Rescore</Button>
                    {delta !== null ? <Text size="1" color={delta >= 0 ? "green" : "red"}>Δ {fmtReward(delta)}</Text> : null}
                  </Flex>
                ) : null
              }
              flush
            >
              {reward ? (
                <table className="data-table">
                  <thead><tr><th>signal</th><th className="num">value</th><th className="num">weight</th><th>reason</th></tr></thead>
                  <tbody>
                    {reward.signals.map((s) => (
                      <tr key={s.name}><td className="mono">{s.name}</td><td className="num">{s.value.toFixed(3)}</td><td className="num">{s.weight.toFixed(2)}</td><td>{s.reason}</td></tr>
                    ))}
                    {reward.audit_flags.length ? <tr><td colSpan={4} className="muted">audit: {reward.audit_flags.join(", ")}</td></tr> : null}
                  </tbody>
                </table>
              ) : <div className="empty">reward is computed when the rollout ends</div>}
            </Panel>
          </Tabs.Content>
          <Tabs.Content value="events">
            <Panel title="Event log" right="durable, sequenced">
              <EventTimeline events={events.data ?? []} />
            </Panel>
          </Tabs.Content>
          <Tabs.Content value="task">
            <Panel title="Task">
              <dl className="kv">
                <dt>task id</dt><dd>{task.id}</dd>
                <dt>repo snapshot</dt><dd>{task.repo_snapshot}</dd>
                <dt>horizon</dt><dd>{task.horizon}</dd>
                <dt>success criteria</dt><dd>{task.success_criteria.join(", ") || "(none)"}</dd>
                <dt>adapter</dt><dd>{m.adapter_id ?? "(base model)"}</dd>
                <dt>created</dt><dd>{fmtDateTime(m.created_at)}</dd>
                <dt>started</dt><dd>{fmtDateTime(m.started_at)}</dd>
                <dt>finished</dt><dd>{fmtDateTime(m.finished_at)}</dd>
              </dl>
            </Panel>
          </Tabs.Content>
        </div>
      </Tabs.Root>
    </>
  );
}
