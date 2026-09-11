import { Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { LineChart, type ChartSeries } from "../components/LineChart";
import { createRun, createTrainingRun, fetchBudget, fetchRuntimeConfig } from "../lib/api";
import { useLiveDashboard, type RunProgress } from "../lib/store";
import type { BudgetStatus, RunManifest, RuntimeConfig } from "../lib/types";

export function DashboardPage() {
  const { snapshot, logs, progress, status, error } = useLiveDashboard();
  const [budget, setBudget] = useState<BudgetStatus | null>(null);
  const [config, setConfig] = useState<RuntimeConfig | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [training, setTraining] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("Explore this repository and summarise what it does.");
  const [launching, setLaunching] = useState(false);
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
    const timer = window.setInterval(() => void load(), 10_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    let active = true;
    fetchRuntimeConfig()
      .then((c) => active && setConfig(c))
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, []);

  const scored = useMemo(
    () =>
      (snapshot?.runs ?? [])
        .filter((r) => r.terminal_reward !== null && r.terminal_reward !== undefined)
        .sort((a, b) => Date.parse(a.created_at) - Date.parse(b.created_at)),
    [snapshot],
  );

  const rewardSeries = useMemo<ChartSeries[]>(
    () => [
      {
        name: "terminal reward",
        color: "#0e6a38",
        points: scored.map((r, i) => ({ x: i + 1, y: r.terminal_reward ?? 0 })),
      },
    ],
    [scored],
  );

  const costSeries = useMemo<ChartSeries[]>(() => {
    let running = 0;
    return [
      {
        name: "cumulative $",
        color: "#244aa5",
        points: scored.map((r, i) => {
          running += r.cost_usd;
          return { x: i + 1, y: running };
        }),
      },
    ];
  }, [scored]);

  const toggleSelected = (runId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(runId)) next.delete(runId);
      else next.add(runId);
      return next;
    });
  };

  const launch = async () => {
    if (!prompt.trim()) return;
    setLaunching(true);
    setActionError(null);
    try {
      await createRun({ prompt: prompt.trim(), repo_snapshot: ".", infra_target: "local", success_criteria: [] });
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setLaunching(false);
    }
  };

  const startTraining = async () => {
    if (selected.size === 0) return;
    setTraining(true);
    setActionError(null);
    try {
      const result = await createTrainingRun({ sample_run_ids: Array.from(selected) });
      setSelected(new Set());
      navigate({ to: "/training/$trainingRunId", params: { trainingRunId: result.training_run_id } });
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setTraining(false);
    }
  };

  if (error && !snapshot) return <section className="panel">Dashboard error: {error}</section>;
  if (!snapshot) return <section className="panel">Loading dashboard...</section>;

  const jobs = snapshot.jobs;

  return (
    <div className="grid">
      <section className="panel panel-wide">
        <div className="panel-header">
          <h2>Launch rollout</h2>
          <span>
            {config ? (
              <span className="badge backend-pill" title={`policy: ${config.policy_name}`}>
                {config.env_backend}
                {config.env_backend === "verifiers" && config.verifiers_env_id ? ` · ${config.verifiers_env_id}` : ""}
                {" · "}
                {config.policy_name}
                {" · "}
                sandbox {config.sandbox}
              </span>
            ) : null}
          </span>
        </div>
        <div className="launch-bar">
          <input
            className="launch-input"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && void launch()}
            placeholder="Task prompt"
          />
          <button className="action-btn action-btn-neutral" onClick={launch} disabled={launching}>
            {launching ? "Queuing..." : "Queue rollout"}
          </button>
        </div>
        {actionError ? <p className="errors">{actionError}</p> : null}
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Jobs</h2>
          <span>
            <em>{status}</em>
            {config ? ` · ${config.worker_id}` : ""}
          </span>
        </div>
        <div className="job-counts">
          <Count label="queued" value={jobs.queued} />
          <Count label="running" value={jobs.running} />
          <Count label="completed" value={jobs.completed} />
          <Count label="failed" value={jobs.failed} />
          <Count label="cancelled" value={jobs.cancelled} />
        </div>
        {config ? <p className="muted">max parallel rollouts: {config.max_parallel_rollouts}</p> : null}
      </section>

      {budget && budget.cap_usd > 0 ? (
        <section className="panel budget-panel">
          <div className="panel-header">
            <h2>Budget</h2>
            <span>
              ${budget.spent_usd.toFixed(4)} / ${budget.cap_usd.toFixed(2)} · {budget.window_hours}h
            </span>
          </div>
          <div className={`budget-bar ${budget.exceeded ? "budget-bar-over" : ""}`}>
            <div className="budget-fill" style={{ width: `${Math.min(100, (budget.spent_usd / budget.cap_usd) * 100)}%` }} />
          </div>
          {budget.exceeded ? (
            <p className="budget-warn">Cap reached: new rollouts will be rejected.</p>
          ) : budget.remaining_usd !== null && budget.remaining_usd !== undefined ? (
            <p className="muted">${budget.remaining_usd.toFixed(4)} remaining</p>
          ) : null}
        </section>
      ) : null}

      <section className="panel">
        <div className="panel-header">
          <h2>Runs</h2>
          <span>{snapshot.runs.length}</span>
        </div>
        {selected.size > 0 ? (
          <div className="train-bar">
            <span>{selected.size} selected</span>
            <button className="action-btn action-btn-neutral" disabled={training} onClick={startTraining}>
              {training ? "Starting..." : "Train from selection"}
            </button>
            <button className="action-btn" onClick={() => setSelected(new Set())}>
              Clear
            </button>
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
                <p className="muted">{worker.detail}</p>
              </div>
              <div className={`badge badge-${worker.status}`}>{worker.status}</div>
            </div>
          ))}
        </div>
      </section>

      {scored.length > 1 ? (
        <>
          <section className="panel">
            <div className="panel-header">
              <h2>Reward over time</h2>
              <span>{scored.length} scored runs</span>
            </div>
            <LineChart series={rewardSeries} xLabel="run #" yLabel="reward" yDomain={[-1, 1]} />
          </section>
          <section className="panel">
            <div className="panel-header">
              <h2>Cumulative cost</h2>
              <span>${scored.reduce((acc, r) => acc + r.cost_usd, 0).toFixed(4)}</span>
            </div>
            <LineChart series={costSeries} xLabel="run #" yLabel="$" />
          </section>
        </>
      ) : null}

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
              <div key={line.seq} className="log-row">
                <span className={`badge badge-${line.payload.level.toLowerCase()}`}>{line.payload.level}</span>
                <span className="log-time">{new Date(line.at).toLocaleTimeString()}</span>
                <span className="log-logger">{line.payload.logger}</span>
                <span className="log-run">{line.subject?.kind === "run" ? line.subject.id : ""}</span>
                <span className="log-message">{line.payload.message}</span>
              </div>
            ))}
        </div>
      </section>
    </div>
  );
}

function Count({ label, value }: { label: string; value: number }) {
  return (
    <div className={`job-count job-count-${label}`}>
      <strong>{value}</strong>
      <span>{label}</span>
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
  progress: RunProgress | undefined;
  selected: boolean;
  selectable: boolean;
  onToggleSelect: () => void;
}) {
  const live = progress && (run.status === "running" || run.status === "queued");
  return (
    <div className={`run-card ${selected ? "run-card-selected" : ""}`}>
      <div className="run-title">
        <label className="run-select" onClick={(e) => e.stopPropagation()}>
          <input type="checkbox" checked={selected} disabled={!selectable} onChange={onToggleSelect} />
          <Link to="/runs/$runId" params={{ runId: run.id }}>
            <strong>{run.id}</strong>
          </Link>
        </label>
        <span className={`badge badge-${run.status}`}>{run.status}</span>
      </div>
      <p className="run-model">
        {run.model_id} · profile {run.profile}
      </p>
      <p>
        {run.infra_target} · horizon {run.horizon} · {run.step_count} steps · {run.tokens.toLocaleString()} tok · $
        {run.cost_usd.toFixed(4)}
        {run.terminal_reward !== null && run.terminal_reward !== undefined ? ` · reward ${run.terminal_reward.toFixed(3)}` : ""}
        {run.adapter_id ? ` · ${run.adapter_id}` : ""}
      </p>
      {live ? (
        <p className="run-progress">
          turn {progress.turn + 1}/{run.horizon}
          {progress.tool ? ` · ${progress.tool}` : ""}
          {" · "}${progress.cost_usd.toFixed(4)} · {progress.tokens.toLocaleString()} tok
        </p>
      ) : null}
      {run.error ? <p className="errors">{run.error}</p> : null}
    </div>
  );
}
