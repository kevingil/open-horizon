import { Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { fetchDashboard } from "../lib/api";
import type { DashboardSnapshot } from "../lib/types";

export function DashboardPage() {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const next = await fetchDashboard();
        if (active) {
          setSnapshot(next);
        }
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : "Unknown error");
        }
      }
    }

    void load();
    const timer = window.setInterval(() => {
      void load();
    }, 4000);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  if (error) {
    return <section className="panel">Dashboard error: {error}</section>;
  }

  if (!snapshot) {
    return <section className="panel">Loading dashboard...</section>;
  }

  return (
    <div className="grid">
      <section className="panel">
        <div className="panel-header">
          <h2>Runs</h2>
          <span>{snapshot.runs.length}</span>
        </div>
        <div className="stack">
          {snapshot.runs.map((run) => (
            <Link key={run.id} to="/runs/$runId" params={{ runId: run.id }} className="run-card">
              <div className="run-title">
                <strong>{run.id}</strong>
                <span className={`badge badge-${run.status}`}>{run.status}</span>
              </div>
              <p>{run.model_id}</p>
              <p>
                {run.infra_target} · ${run.estimated_cost_usd.toFixed(2)}
              </p>
            </Link>
          ))}
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
    </div>
  );
}
