import { Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { LineChart, type ChartSeries } from "../components/LineChart";
import { createTrainingRun, fetchBudget, type BudgetStatus } from "../lib/api";
import { useLiveDashboard } from "../lib/store";
import type { RunManifest } from "../lib/types";

export function DashboardPage() {
  const { snapshot, logs, progress, status, error } = useLiveDashboard();
  const [budget, setBudget] = useState<BudgetStatus | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [training, setTraining] = useState(false);
  const [trainError, setTrainError] = useState<string | null>(null);
  const navigate = useNavigate();

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

  const rewardSeries = useMemo<ChartSeries[]>(() => {
    if (!snapshot) return [];
    // Build a time-ordered series of terminal rewards from manifests; we use
    // RunDetail when available via the WebSocket completed event, but the
    // dashboard snapshot only has manifests, so cost is the only signal here.
    // Show estimated_cost_usd as a proxy on the runs panel; for *reward over
    // time* we render whatever runs exist (sorted oldest -> newest by
    // created_at) once we have detail. The detail array isn't on the
    // snapshot, so we punt on per-run reward here and instead render cost.
    const ordered = [...snapshot.runs].sort(
      (a, b) => Date.parse(a.created_at) - Date.parse(b.created_at),
    );
    const points = ordered.map((r, i) => ({ x: i, y: r.estimated_cost_usd }));
    return [{ name: "cost / run", color: "#244aa5", points }];
  }, [snapshot]);

  const toggleSelected = (runId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(runId)) next.delete(runId);
      else next.add(runId);
      return next;
    });
  };

  const startTraining = async () => {
    if (selected.size === 0) return;
    setTraining(true);
    setTrainError(null);
    try {
      const result = await createTrainingRun({
        sample_run_ids: Array.from(selected),
      });
      setSelected(new Set());
      navigate({ to: "/training/$trainingRunId", params: { trainingRunId: result.training_run_id } });
    } catch (err) {
      setTrainError(err instanceof Error ? err.message : String(err));
    } finally {
      setTraining(false);
    }
  };

  if (error && !snapshot) {
    return <section className="panel">Dashboard error: {error}</section>;
  }
  if (!snapshot) {
    return <section className="panel">Loading dashboard...</section>;
  }

  const completed = snapshot.runs.filter((r) => r.status === "completed");

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
        {selected.size > 0 ? (
          <div className="train-bar">
            <span>{selected.size} selected</span>
            <button
              className="action-btn action-btn-neutral"
              disabled={training}
              onClick={startTraining}
            >
              {training ? "Starting..." : "Train from selection"}
            </button>
            <button className="action-btn" onClick={() => setSelected(new Set())}>
              Clear
            </button>
            {trainError ? <span className="errors">{trainError}</span> : null}
          </div>
        ) : null}
        <div className="stack">
          {snapshot.runs.map((run) => (
            <RunCard
              key={run.id}
              run={run}
              progress={progress[run.id]}
              selected={selected.has(run.id)}
              selectable={run.status === "completed"}
              onToggleSelect={() => toggleSelected(run.id)}
            />
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

      {completed.length > 1 ? (
        <section className="panel panel-wide">
          <div className="panel-header">
            <h2>Cost over time</h2>
            <span>{completed.length} runs</span>
          </div>
          <LineChart series={rewardSeries} xLabel="run #" yLabel="$" />
        </section>
      ) : null}

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

function RunCard({
  run,
  progress,
  selected,
  selectable,
  onToggleSelect,
}: {
  run: RunManifest;
  progress: { step_index: number; tool: string | null; tokens: number; cost_usd: number } | undefined;
  selected: boolean;
  selectable: boolean;
  onToggleSelect: () => void;
}) {
  return (
    <div className={`run-card ${selected ? "run-card-selected" : ""}`}>
      <div className="run-title">
        <label className="run-select" onClick={(e) => e.stopPropagation()}>
          <input
            type="checkbox"
            checked={selected}
            disabled={!selectable}
            onChange={onToggleSelect}
          />
          <Link to="/runs/$runId" params={{ runId: run.id }}>
            <strong>{run.id}</strong>
          </Link>
        </label>
        <span className={`badge badge-${run.status}`}>{run.status}</span>
      </div>
      <p className="run-model">{run.model_id}</p>
      <p>
        {run.infra_target} · ${run.estimated_cost_usd.toFixed(4)}
        {run.adapter_id ? ` · ${run.adapter_id}` : ""}
      </p>
      {progress ? (
        <p className="run-progress">
          step {progress.step_index + 1}
          {progress.tool ? ` · ${progress.tool}` : ""}
          {" · "}${progress.cost_usd.toFixed(4)} · {progress.tokens.toLocaleString()} tok
        </p>
      ) : null}
    </div>
  );
}
