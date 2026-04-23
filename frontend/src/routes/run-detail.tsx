import { useParams } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { fetchRun } from "../lib/api";
import type { RunDetail, TrajectoryStep } from "../lib/types";

interface RewardProvenance {
  rubric?: string;
  signals?: { name: string; value: number; weight: number; reason: string }[];
}

export function RunDetailPage() {
  const { runId } = useParams({ from: "/runs/$runId" });
  const [run, setRun] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const next = await fetchRun(runId);
        if (active) {
          setRun(next);
        }
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : "Unknown error");
        }
      }
    }
    void load();
    return () => {
      active = false;
    };
  }, [runId]);

  const provenance = useMemo<RewardProvenance | null>(() => {
    if (!run) return null;
    try {
      return JSON.parse(run.reward.provenance) as RewardProvenance;
    } catch {
      return null;
    }
  }, [run]);

  if (error) {
    return <section className="panel">Run error: {error}</section>;
  }
  if (!run) {
    return <section className="panel">Loading run detail...</section>;
  }

  return (
    <div className="grid">
      <section className="panel">
        <div className="panel-header">
          <h2>{run.manifest.id}</h2>
          <span className={`badge badge-${run.manifest.status}`}>{run.manifest.status}</span>
        </div>
        <p>{run.task.prompt}</p>
        <dl className="kv">
          <dt>Model</dt>
          <dd>{run.manifest.model_id}</dd>
          <dt>Infra</dt>
          <dd>{run.manifest.infra_target}</dd>
          <dt>Cost</dt>
          <dd>${run.manifest.estimated_cost_usd.toFixed(4)}</dd>
          {run.manifest.adapter_id ? (
            <>
              <dt>Adapter</dt>
              <dd>{run.manifest.adapter_id}</dd>
            </>
          ) : null}
        </dl>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Reward</h2>
          <span>{run.reward.terminal_reward.toFixed(3)}</span>
        </div>
        {provenance?.rubric ? <p>Rubric: {provenance.rubric}</p> : null}
        <p>Audit flags: {run.reward.audit_flags.join(", ") || "none"}</p>
        {provenance?.signals ? (
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
              {provenance.signals.map((s) => (
                <tr key={s.name}>
                  <td>{s.name}</td>
                  <td>{s.value.toFixed(3)}</td>
                  <td>{s.weight.toFixed(2)}</td>
                  <td>{s.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted">provenance: {run.reward.provenance}</p>
        )}
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
            <StepRow key={`${step.index}-${step.timestamp}`} step={step} />
          ))}
        </div>
      </section>
    </div>
  );
}

function StepRow({ step }: { step: TrajectoryStep }) {
  const parsed = useMemo(() => tryParse(step.content), [step.content]);
  const toolName = typeof parsed?.tool === "string"
    ? parsed.tool
    : typeof parsed?.name === "string"
      ? parsed.name
      : null;
  return (
    <div className="row-card step-row">
      <div className="step-main">
        <div className="step-head">
          <strong>
            {step.index} · {step.actor}
          </strong>
          {toolName ? <span className="tool-chip">{toolName}</span> : null}
          <span className="artifact-kind">{step.kind}</span>
        </div>
        {parsed ? (
          <pre className="step-json">{JSON.stringify(parsed, null, 2)}</pre>
        ) : (
          <p className="step-text">{step.content}</p>
        )}
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
