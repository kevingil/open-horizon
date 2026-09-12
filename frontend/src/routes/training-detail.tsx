import { Flex, Grid, Text } from "@radix-ui/themes";
import { useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "@tanstack/react-router";
import { useCallback, useMemo } from "react";
import { Panel } from "../components/Panel";
import { StatTile } from "../components/StatTile";
import { StatusBadge } from "../components/StatusBadge";
import { TimeSeries } from "../components/TimeSeries";
import type { EventEnvelope } from "../lib/events";
import { fmtDateTime, fmtDuration } from "../lib/format";
import { useEvents } from "../lib/hub";
import { useTrainingRun } from "../lib/query";
import type { TrainingRunRecord } from "../lib/types";

export function TrainingRunDetailPage() {
  const { trainingRunId } = useParams({ from: "/training/$trainingRunId" });
  const qc = useQueryClient();
  const run = useTrainingRun(trainingRunId);

  const onEvent = useCallback(
    (e: EventEnvelope) => {
      if (e.subject?.kind !== "training_run" || e.subject.id !== trainingRunId || e.kind !== "training.metric") return;
      const metric = e.payload.metric;
      qc.setQueryData<TrainingRunRecord>(["training-run", trainingRunId], (prev) => {
        if (!prev || prev.metrics.some((m) => m.step === metric.step)) return prev;
        return { ...prev, metrics: [...prev.metrics, metric].sort((a, b) => a.step - b.step) };
      });
    },
    [qc, trainingRunId],
  );
  useEvents(onEvent);

  const metrics = run.data?.metrics ?? [];
  const x = useMemo(() => metrics.map((m) => m.step), [metrics]);
  const loss = useMemo(() => [{ label: "loss", values: metrics.map((m) => m.loss), slot: 8 as const }], [metrics]);
  const reward = useMemo(() => [{ label: "mean reward", values: metrics.map((m) => m.mean_reward ?? null), slot: 6 as const }], [metrics]);
  const kl = useMemo(() => [{ label: "kl", values: metrics.map((m) => m.kl ?? null), slot: 1 as const }], [metrics]);
  const rewardRange = useMemo<[number, number]>(() => [0, 1], []);
  const stepFmt = useCallback((v: number) => v.toFixed(v < 1 ? 3 : 2), []);

  if (run.error) return <div className="empty">{String(run.error)}</div>;
  if (!run.data) return <div className="empty">loading…</div>;
  const r = run.data;
  return (
    <>
      <Flex align="center" gap="3" wrap="wrap">
        <Link to="/training" className="muted" style={{ fontSize: 12 }}>← training</Link>
        <Text size="4" weight="bold" className="mono">{r.id}</Text>
        <StatusBadge status={r.status} />
        <Text size="1" className="muted">{r.trainer}</Text>
      </Flex>
      <div className="tiles">
        <StatTile label="Steps" value={metrics.length} sub={`hyperparams ${JSON.stringify(r.hyperparams)}`} />
        <StatTile label="Final loss" value={metrics.at(-1)?.loss.toFixed(4) ?? "–"} sub={metrics[0] ? `from ${metrics[0].loss.toFixed(4)}` : ""} />
        <StatTile label="Samples" value={r.sample_run_ids.length} sub="completed rollouts" />
        <StatTile label="Duration" value={fmtDuration(r.created_at, r.status === "running" || r.status === "queued" ? null : r.updated_at)} sub={fmtDateTime(r.created_at)} />
        <StatTile label="Adapter" value={<span className="mono" style={{ fontSize: 14 }}>{r.adapter_out ?? "–"}</span>} sub={r.adapter_in ? `parent ${r.adapter_in}` : "from base model"} />
      </div>
      {r.error ? <Text size="2" color="red">error: {r.error}</Text> : null}
      <Grid columns={{ initial: "1", md: "3" }} gap="3">
        <Panel title="Loss"><TimeSeries x={x} series={loss} yLabel="loss" format={stepFmt} timeAxis={false} /></Panel>
        <Panel title="Mean reward"><TimeSeries x={x} series={reward} yLabel="reward" yRange={rewardRange} format={stepFmt} timeAxis={false} /></Panel>
        <Panel title="KL"><TimeSeries x={x} series={kl} yLabel="kl" format={stepFmt} timeAxis={false} /></Panel>
      </Grid>
      <Panel title="Sample rollouts" right={`${r.sample_run_ids.length}`} flush>
        <table className="data-table">
          <tbody>
            {r.sample_run_ids.map((id) => (
              <tr key={id}><td><Link to="/rollouts/$runId" params={{ runId: id }} className="mono">{id}</Link></td></tr>
            ))}
            {r.sample_run_ids.length === 0 ? <tr><td className="empty">no samples</td></tr> : null}
          </tbody>
        </table>
      </Panel>
    </>
  );
}
