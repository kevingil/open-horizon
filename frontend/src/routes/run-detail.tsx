import { useParams } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { cancelRun, fetchRubrics, fetchRun, rescoreRun } from "../lib/api";
import { runIdOf } from "../lib/events";
import { useEventStream } from "../lib/socket";
import type { RubricInfo, RunDetail, TrajectoryStep } from "../lib/types";

export function RunDetailPage() {
  const { runId } = useParams({ from: "/runs/$runId" });
  const [run, setRun] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [rubrics, setRubrics] = useState<RubricInfo[]>([]);
  const [selectedRubric, setSelectedRubric] = useState<string>("");
  const [rescoring, setRescoring] = useState(false);
  const [rescoreDelta, setRescoreDelta] = useState<number | null>(null);

  const refresh = async () => {
    try {
      setRun(await fetchRun(runId));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const [next, rubricList] = await Promise.all([fetchRun(runId), fetchRubrics()]);
        if (!active) return;
        setRun(next);
        setRubrics(rubricList);
        setSelectedRubric((cur) => cur || rubricList[0]?.name || "");
      } catch (err) {
        if (active) setError(err instanceof Error ? err.message : "Unknown error");
      }
    }
    void load();
    return () => {
      active = false;
    };
  }, [runId]);

  // Live: append steps as they land, refetch on terminal events.
  useEventStream((event) => {
    if (runIdOf(event) !== runId) return;
    if (event.kind === "step.recorded") {
      const step = event.payload.step;
      setRun((prev) => {
        if (!prev) return prev;
        if (prev.trajectory.steps.some((s) => s.index === step.index)) return prev;
        const steps = [...prev.trajectory.steps, step].sort((a, b) => a.index - b.index);
        return { ...prev, trajectory: { ...prev.trajectory, steps }, manifest: { ...prev.manifest, step_count: steps.length } };
      });
    }
    if (event.kind === "rollout.started" || event.kind === "rollout.queued") {
      setRun((prev) => (prev ? { ...prev, manifest: event.payload.manifest } : prev));
    }
    if (
      event.kind === "rollout.completed" ||
      event.kind === "rollout.failed" ||
      event.kind === "rollout.cancelled" ||
      event.kind === "reward.computed"
    ) {
      void refresh();
    }
  });

  const onCancel = async () => {
    setCancelling(true);
    try {
      await cancelRun(runId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setCancelling(false);
    }
  };

  const onRescore = async () => {
    if (!selectedRubric) return;
    setRescoring(true);
    try {
      const result = await rescoreRun(runId, selectedRubric);
      setRescoreDelta(result.delta ?? null);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRescoring(false);
    }
  };

  if (error) return <section className="panel">Run error: {error}</section>;
  if (!run) return <section className="panel">Loading run detail...</section>;

  const m = run.manifest;
  const reward = run.reward ?? null;
  const active = m.status === "running" || m.status === "queued";

  return (
    <div className="grid">
      <section className="panel">
        <div className="panel-header">
          <h2>{m.id}</h2>
          <span className={`badge badge-${m.status}`}>{m.status}</span>
        </div>
        <p>{run.task.prompt}</p>
        {active ? (
          <button className="action-btn" onClick={onCancel} disabled={cancelling}>
            {cancelling ? "Cancelling..." : "Cancel rollout"}
          </button>
        ) : null}
        <dl className="kv">
          <dt>Model</dt>
          <dd>{m.model_id}</dd>
          <dt>Profile</dt>
          <dd>{m.profile}</dd>
          <dt>Infra</dt>
          <dd>{m.infra_target}</dd>
          <dt>Horizon</dt>
          <dd>{m.horizon}</dd>
          <dt>Tokens</dt>
          <dd>{m.tokens.toLocaleString()}</dd>
          <dt>Cost</dt>
          <dd>${m.cost_usd.toFixed(4)}</dd>
          {m.adapter_id ? (
            <>
              <dt>Adapter</dt>
              <dd>{m.adapter_id}</dd>
            </>
          ) : null}
          {run.task.success_criteria.length ? (
            <>
              <dt>Criteria</dt>
              <dd>{run.task.success_criteria.join(", ")}</dd>
            </>
          ) : null}
        </dl>
        {m.error ? <p className="errors">error: {m.error}</p> : null}
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Reward</h2>
          <span>{reward ? reward.terminal_reward.toFixed(3) : "unscored"}</span>
        </div>
        {reward ? (
          <p>
            Rubric: {reward.rubric}
            <span className="badge"> {reward.source}</span>
          </p>
        ) : null}
        {reward ? <p>Audit flags: {reward.audit_flags.join(", ") || "none"}</p> : null}
        {rubrics.length > 0 && !active ? (
          <div className="rescore">
            <label>
              Rescore with
              <select value={selectedRubric} onChange={(e) => setSelectedRubric(e.target.value)} disabled={rescoring}>
                {rubrics.map((r) => (
                  <option key={r.name} value={r.name}>
                    {r.name}
                  </option>
                ))}
              </select>
            </label>
            <button className="action-btn action-btn-neutral" onClick={onRescore} disabled={rescoring}>
              {rescoring ? "Scoring..." : "Apply"}
            </button>
            {rescoreDelta !== null ? (
              <span className={rescoreDelta >= 0 ? "delta-pos" : "delta-neg"}>
                Δ {rescoreDelta >= 0 ? "+" : ""}
                {rescoreDelta.toFixed(3)}
              </span>
            ) : null}
          </div>
        ) : null}
        {reward && reward.signals.length ? (
          <table className="signal-table">
            <thead>
              <tr>
                <th>signal</th>
                <th>value</th>
                <th>weight</th>
                <th>reason</th>
              </tr>
            </thead>
            <tbody>
              {reward.signals.map((s) => (
                <tr key={s.name}>
                  <td>{s.name}</td>
                  <td>{s.value.toFixed(3)}</td>
                  <td>{s.weight.toFixed(2)}</td>
                  <td>{s.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </section>

      <section className="panel panel-wide">
        <div className="panel-header">
          <h2>Trajectory</h2>
          <span>{run.trajectory.steps.length} steps</span>
        </div>
        {run.trajectory.errors.length ? (
          <div className="errors">
            {run.trajectory.errors.map((e, i) => (
              <p key={i}>error: {e}</p>
            ))}
          </div>
        ) : null}
        <div className="stack">
          {run.trajectory.steps.map((step) => (
            <StepRow key={step.index} step={step} />
          ))}
        </div>
      </section>
    </div>
  );
}

function StepRow({ step }: { step: TrajectoryStep }) {
  const parsed = useMemo(() => tryParse(step.content), [step.content]);
  const toolName = typeof parsed?.tool === "string" ? parsed.tool : null;
  return (
    <div className="row-card step-row">
      <div className="step-main">
        <div className="step-head">
          <strong>
            {step.index} · {step.actor}
          </strong>
          {toolName ? <span className="tool-chip">{toolName}</span> : null}
          <span className="artifact-kind">{step.kind}</span>
          <span className="log-time">{new Date(step.at).toLocaleTimeString()}</span>
        </div>
        {parsed ? <pre className="step-json">{JSON.stringify(parsed, null, 2)}</pre> : <p className="step-text">{step.content}</p>}
      </div>
    </div>
  );
}

function tryParse(text: string): Record<string, unknown> | null {
  try {
    const v = JSON.parse(text);
    return typeof v === "object" && v !== null ? (v as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}
