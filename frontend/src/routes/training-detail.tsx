import { Link, useParams } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { LineChart, type ChartSeries } from "../components/LineChart";
import { fetchTrainingRun } from "../lib/api";
import { subjectId } from "../lib/events";
import { useEventStream } from "../lib/socket";
import type { TrainingMetricPoint, TrainingRunRecord } from "../lib/types";

export function TrainingRunDetailPage() {
  const { trainingRunId } = useParams({ from: "/training/$trainingRunId" });
  const [record, setRecord] = useState<TrainingRunRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [liveMetrics, setLiveMetrics] = useState<TrainingMetricPoint[]>([]);
  const [terminalKind, setTerminalKind] = useState<string | null>(null);

  // Initial fetch + polling fallback so a refresh on a completed run still works.
  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const got = await fetchTrainingRun(trainingRunId);
        if (active) setRecord(got);
      } catch (err) {
        if (active) setError(err instanceof Error ? err.message : String(err));
      }
    }
    void load();
    const t = window.setInterval(() => void load(), 5000);
    return () => {
      active = false;
      window.clearInterval(t);
    };
  }, [trainingRunId]);

  // Subscribe to live training events for this run.
  const status = useEventStream((event) => {
    if (subjectId(event, "training_run") !== trainingRunId) return;
    if (event.kind === "training.metric") {
      setLiveMetrics((prev) => [...prev, event.payload.metric]);
    }
    if (event.kind === "training.started") {
      setRecord(event.payload.record);
    }
    if (event.kind === "training.completed" || event.kind === "training.failed") {
      setTerminalKind(event.kind);
      setRecord(event.payload.record);
    }
  });

  const metrics = useMemo<TrainingMetricPoint[]>(() => {
    // Prefer the persisted metrics if completed; otherwise overlay live ticks
    // on top of whatever the API returned (covers a mid-flight page load).
    const persisted = record?.metrics ?? [];
    if (record?.status === "completed" || record?.status === "failed") return persisted;
    const seen = new Set(persisted.map((p) => p.step));
    const merged = [...persisted, ...liveMetrics.filter((p) => !seen.has(p.step))];
    merged.sort((a, b) => a.step - b.step);
    return merged;
  }, [record, liveMetrics]);

  const lossSeries: ChartSeries = {
    name: "loss",
    color: "#a62b1f",
    points: metrics.map((m) => ({ x: m.step, y: m.loss })),
  };
  const rewardSeries: ChartSeries = {
    name: "mean_reward",
    color: "#0e6a38",
    points: metrics
      .filter((m): m is TrainingMetricPoint & { mean_reward: number } => typeof m.mean_reward === "number")
      .map((m) => ({ x: m.step, y: m.mean_reward })),
  };
  const klSeries: ChartSeries = {
    name: "kl",
    color: "#244aa5",
    points: metrics
      .filter((m): m is TrainingMetricPoint & { kl: number } => typeof m.kl === "number")
      .map((m) => ({ x: m.step, y: m.kl })),
  };

  if (error && !record) {
    return <section className="panel">Training run error: {error}</section>;
  }
  if (!record) return <section className="panel">Loading training run...</section>;

  return (
    <div className="grid">
      <section className="panel">
        <div className="panel-header">
          <h2>{record.id}</h2>
          <span className={`badge badge-${record.status}`}>{record.status}</span>
        </div>
        <dl className="kv">
          <dt>Parent</dt>
          <dd>
            {record.adapter_in ? (
              <Link to="/adapters">{record.adapter_in}</Link>
            ) : (
              "-"
            )}
          </dd>
          <dt>Child</dt>
          <dd>
            {record.adapter_out ? (
              <Link to="/adapters">{record.adapter_out}</Link>
            ) : (
              "-"
            )}
          </dd>
          <dt>Trainer</dt>
          <dd>{record.trainer}</dd>
          <dt>Samples</dt>
          <dd>{record.sample_run_ids.length}</dd>
          <dt>Steps</dt>
          <dd>{metrics.length}</dd>
          <dt>WS</dt>
          <dd>{status}</dd>
        </dl>
        {Object.keys(record.hyperparams).length > 0 ? (
          <>
            <h3 style={{ marginTop: 12 }}>Hyperparams</h3>
            <pre className="step-json">{JSON.stringify(record.hyperparams, null, 2)}</pre>
          </>
        ) : null}
        {(record.error || (terminalKind === "training.failed" && error)) ? (
          <p className="errors">error: {record.error || error}</p>
        ) : null}
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Loss</h2>
          <span>{lossSeries.points.length} pts</span>
        </div>
        <LineChart series={[lossSeries]} xLabel="step" yLabel="loss" />
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Mean Reward</h2>
          <span>{rewardSeries.points.length} pts</span>
        </div>
        <LineChart
          series={[rewardSeries]}
          xLabel="step"
          yLabel="reward"
          yDomain={[0, 1]}
        />
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>KL</h2>
          <span>{klSeries.points.length} pts</span>
        </div>
        <LineChart series={[klSeries]} xLabel="step" yLabel="kl" />
      </section>

      <section className="panel panel-wide">
        <div className="panel-header">
          <h2>Sample Rollouts</h2>
          <span>{record.sample_run_ids.length}</span>
        </div>
        <div className="stack">
          {record.sample_run_ids.map((id) => (
            <Link
              key={id}
              to="/runs/$runId"
              params={{ runId: id }}
              className="run-card"
            >
              <strong>{id}</strong>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
