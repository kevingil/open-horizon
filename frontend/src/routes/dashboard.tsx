import { Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { fetchBudget, type BudgetStatus } from "../lib/api";
import { useLiveDashboard } from "../lib/store";

export function DashboardPage() {
  const { snapshot, logs, progress, status, error } = useLiveDashboard();
  const [budget, setBudget] = useState<BudgetStatus | null>(null);

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const b = await fetchBudget();
        if (active) setBudget(b);
      } catch {
        // non-fatal; panel just won't render
      }
    }
    void load();
    const timer = window.setInterval(() => void load(), 15_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  if (error && !snapshot) {
    return <section className="panel">Dashboard error: {error}</section>;
  }

  if (!snapshot) {
    return <section className="panel">Loading dashboard...</section>;
  }

  return (
    <div className="grid">
      {budget && budget.cap_usd > 0 ? (
        <section className="panel panel-wide budget-panel">
          <div className="panel-header">
            <h2>Budget</h2>
            <span>
              ${budget.spent_usd.toFixed(4)} / ${budget.cap_usd.toFixed(2)} ·{" "}
              {budget.window_hours}h
            </span>
          </div>
          <div className={`budget-bar ${budget.exceeded ? "budget-bar-over" : ""}`}>
            <div
              className="budget-fill"
              style={{ width: `${Math.min(100, (budget.spent_usd / budget.cap_usd) * 100)}%` }}
            />
          </div>
          {budget.exceeded ? (
            <p className="budget-warn">Cap reached - new rollouts will be rejected.</p>
          ) : budget.remaining_usd !== null ? (
            <p className="muted">${budget.remaining_usd.toFixed(4)} remaining</p>
          ) : null}
        </section>
      ) : null}

      <section className="panel">
        <div className="panel-header">
          <h2>Runs</h2>
          <span>
            {snapshot.runs.length} · <em>{status}</em>
          </span>
        </div>
        <div className="stack">
          {snapshot.runs.map((run) => {
            const p = progress[run.id];
            return (
              <Link key={run.id} to="/runs/$runId" params={{ runId: run.id }} className="run-card">
                <div className="run-title">
                  <strong>{run.id}</strong>
                  <span className={`badge badge-${run.status}`}>{run.status}</span>
                </div>
                <p className="run-model">{run.model_id}</p>
                <p>
                  {run.infra_target} · ${run.estimated_cost_usd.toFixed(4)}
                </p>
                {p ? (
                  <p className="run-progress">
                    step {p.step_index + 1}
                    {p.tool ? ` · ${p.tool}` : ""}
                    {" · "}${p.cost_usd.toFixed(4)} · {p.tokens.toLocaleString()} tok
                  </p>
                ) : null}
              </Link>
            );
          })}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Workers</h2>
          <span>{snapshot.workers.length}</span>
        </div>
        <div className="stack">
          {snapshot.workers.map((worker) => (
            <div key={worker.id} className="row-card">
              <div>
                <strong>{worker.role}</strong>
                <p>{worker.id}</p>
              </div>
              <div className={`badge badge-${worker.status}`}>{worker.status}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="panel panel-wide">
        <div className="panel-header">
          <h2>Recent Artifacts</h2>
          <span>{snapshot.recent_artifacts.length}</span>
        </div>
        <div className="stack">
          {snapshot.recent_artifacts.map((artifact) => (
            <div key={`${artifact.kind}-${artifact.path}`} className="row-card">
              <div>
                <strong>{artifact.name}</strong>
                <p>{artifact.path}</p>
              </div>
              <div className="artifact-kind">{artifact.kind}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="panel panel-wide">
        <div className="panel-header">
          <h2>Live Logs</h2>
          <span>{logs.length}</span>
        </div>
        <div className="stack log-stack">
          {logs
            .slice()
            .reverse()
            .map((line) => (
              <div key={line.event_id} className="log-row">
                <span className={`badge badge-${line.level.toLowerCase()}`}>{line.level}</span>
                <span className="log-time">{new Date(line.at).toLocaleTimeString()}</span>
                <span className="log-logger">{line.logger}</span>
                {line.run_id ? <span className="log-run">{line.run_id}</span> : null}
                <span className="log-message">{line.message}</span>
              </div>
            ))}
        </div>
      </section>
    </div>
  );
}
