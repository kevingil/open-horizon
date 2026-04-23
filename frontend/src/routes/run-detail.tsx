import { useParams } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { fetchRun } from "../lib/api";
import type { RunDetail } from "../lib/types";


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
        <p>Infra: {run.manifest.infra_target}</p>
        <p>Cost: ${run.manifest.estimated_cost_usd.toFixed(2)}</p>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Reward</h2>
          <span>{run.reward.terminal_reward.toFixed(2)}</span>
        </div>
        <p>Provenance: {run.reward.provenance}</p>
        <p>Audit flags: {run.reward.audit_flags.join(", ") || "none"}</p>
      </section>

      <section className="panel panel-wide">
        <div className="panel-header">
          <h2>Trajectory</h2>
          <span>{run.trajectory.steps.length} steps</span>
        </div>
        <div className="stack">
          {run.trajectory.steps.map((step) => (
            <div key={`${step.index}-${step.timestamp}`} className="row-card">
              <div>
                <strong>
                  {step.index} · {step.actor}
                </strong>
                <p>{step.content}</p>
              </div>
              <div className="artifact-kind">{step.kind}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
