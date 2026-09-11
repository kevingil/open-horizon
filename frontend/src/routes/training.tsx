import { Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { fetchTrainingRuns } from "../lib/api";
import type { TrainingRunRecord } from "../lib/types";

export function TrainingListPage() {
  const [runs, setRuns] = useState<TrainingRunRecord[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const got = await fetchTrainingRuns();
        if (active) setRuns(got);
      } catch (err) {
        if (active) setError(err instanceof Error ? err.message : String(err));
      }
    }
    void load();
    const t = window.setInterval(() => void load(), 4000);
    return () => {
      active = false;
      window.clearInterval(t);
    };
  }, []);

  if (error) return <section className="panel">Training list error: {error}</section>;

  return (
    <div className="grid">
      <section className="panel panel-wide">
        <div className="panel-header">
          <h2>Training Runs</h2>
          <span>{runs.length}</span>
        </div>
        {runs.length === 0 ? (
          <p className="muted">
            No training runs yet. Trigger one via{" "}
            <code>POST /api/training-runs</code> or <code>horizon train</code>.
          </p>
        ) : (
          <div className="stack">
            {runs.map((r) => (
              <Link
                key={r.id}
                to="/training/$trainingRunId"
                params={{ trainingRunId: r.id }}
                className="run-card"
              >
                <div className="run-title">
                  <strong>{r.id}</strong>
                  <span className={`badge badge-${r.status}`}>{r.status}</span>
                </div>
                <p className="muted">
                  {r.trainer} · parent: {r.adapter_in ?? "-"} · child: {r.adapter_out ?? "-"} ·{" "}
                  {r.metrics.length} steps · {r.sample_run_ids.length} samples
                </p>
                {r.error ? <p className="errors">error: {r.error}</p> : null}
              </Link>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
