import { Flex, Grid, Select, Text } from "@radix-ui/themes";
import { Link } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { Histogram } from "../components/Histogram";
import { Panel } from "../components/Panel";
import { StatTile } from "../components/StatTile";
import { StatusBadge } from "../components/StatusBadge";
import { TimeSeries } from "../components/TimeSeries";
import { fmtAgo, fmtCompact, fmtMs, fmtNumber, fmtPct, fmtReward, fmtUsd } from "../lib/format";
import { useDashboard, useRewardHistogram, useTimeseries } from "../lib/query";
import { LaunchForm } from "./rollouts";

const WINDOWS = [
  { label: "15 min", s: 900, bucket: 30 },
  { label: "1 hour", s: 3600, bucket: 60 },
  { label: "6 hours", s: 21_600, bucket: 300 },
  { label: "24 hours", s: 86_400, bucket: 900 },
];

export function OverviewPage() {
  const [win, setWin] = useState(WINDOWS[0]);
  const dash = useDashboard(win.s);
  const series = useTimeseries(win.s, win.bucket);
  const hist = useRewardHistogram(win.s);

  const x = useMemo(() => (series.data ?? []).map((b) => b.ts), [series.data]);
  const throughput = useMemo(
    () => [
      { label: "completed", values: (series.data ?? []).map((b) => b.completed), slot: 1 as const, bars: true },
      { label: "failed", values: (series.data ?? []).map((b) => b.failed), slot: 8 as const, bars: true },
    ],
    [series.data],
  );
  const tokens = useMemo(() => [{ label: "tokens", values: (series.data ?? []).map((b) => b.tokens), slot: 3 as const }], [series.data]);
  const latency = useMemo(() => [{ label: "p95 policy ms", values: (series.data ?? []).map((b) => b.p95_policy_ms ?? null), slot: 2 as const }], [series.data]);
  const inflight = useMemo(() => [{ label: "in flight", values: (series.data ?? []).map((b) => b.in_flight), slot: 7 as const }], [series.data]);
  const rewardRange = useMemo<[number, number]>(() => [-1, 1], []);
  const reward = useMemo(() => [{ label: "mean reward", values: (series.data ?? []).map((b) => b.mean_reward ?? null), slot: 6 as const }], [series.data]);

  if (dash.error) return <div className="empty">dashboard error: {String(dash.error)}</div>;
  if (!dash.data) return <div className="empty">loading…</div>;
  const s = dash.data.stats;
  const nodesDown = dash.data.nodes.filter((n) => n.status === "down").length;
  const recent = dash.data.runs.slice(0, 8);

  return (
    <>
      <Flex align="center" justify="between">
        <Text size="4" weight="bold">Overview</Text>
        <Flex align="center" gap="2">
          <Text size="1" className="muted">window</Text>
          <Select.Root size="1" value={String(win.s)} onValueChange={(v) => setWin(WINDOWS.find((w) => String(w.s) === v) ?? WINDOWS[0])}>
            <Select.Trigger />
            <Select.Content>
              {WINDOWS.map((w) => <Select.Item key={w.s} value={String(w.s)}>{w.label}</Select.Item>)}
            </Select.Content>
          </Select.Root>
        </Flex>
      </Flex>

      <div className="tiles">
        <StatTile label="In flight" value={s.in_flight} sub={`${s.queued} queued`} />
        <StatTile label="Rollouts / min" value={fmtNumber(s.rollouts_per_min, 2)} sub={`${s.rollouts_completed} done · ${s.rollouts_failed} failed`} />
        <StatTile label="Tokens / s" value={fmtCompact(s.tokens_per_s)} sub={`${fmtUsd(s.cost_per_hour, 3)} / hour`} />
        <StatTile label="Policy p95" value={fmtMs(s.policy_latency.p95_ms)} sub={`p50 ${fmtMs(s.policy_latency.p50_ms)} · ${s.policy_latency.samples} turns`} />
        <StatTile label="Tool p95" value={fmtMs(s.tool_latency.p95_ms)} sub={`p50 ${fmtMs(s.tool_latency.p50_ms)}`} />
        <StatTile label="Mean reward" value={fmtReward(s.mean_reward)} sub="last 50 completed" />
        <StatTile label="Success rate" value={fmtPct(s.success_rate)} sub="terminal runs in window" />
        <StatTile label="Fleet" value={`${dash.data.nodes.length - nodesDown}/${dash.data.nodes.length}`} sub={nodesDown ? `${nodesDown} down` : "all nodes up"} />
      </div>

      <Grid columns={{ initial: "1", md: "2" }} gap="3">
        <Panel title="Rollouts per bucket" right={`${win.bucket}s buckets`}>
          <TimeSeries x={x} series={throughput} yLabel="runs" />
        </Panel>
        <Panel title="Tokens per bucket">
          <TimeSeries x={x} series={tokens} yLabel="tokens" />
        </Panel>
        <Panel title="Policy latency p95" right="per bucket, action steps">
          <TimeSeries x={x} series={latency} yLabel="ms" />
        </Panel>
        <Panel title="In flight">
          <TimeSeries x={x} series={inflight} yLabel="runs" />
        </Panel>
        <Panel title="Mean reward per bucket">
          <TimeSeries x={x} series={reward} yLabel="reward" yRange={rewardRange} />
        </Panel>
        <Panel title="Reward distribution" right="terminal reward, window">
          <Histogram bins={hist.data ?? []} />
        </Panel>
      </Grid>

      <Grid columns={{ initial: "1", md: "3fr 2fr" }} gap="3">
        <Panel title="Recent rollouts" right={<Link to="/rollouts">all rollouts →</Link>} flush>
          <table className="data-table">
            <thead><tr><th>run</th><th>status</th><th>profile</th><th className="num">steps</th><th className="num">tokens</th><th className="num">reward</th><th>updated</th></tr></thead>
            <tbody>
              {recent.map((r) => (
                <tr key={r.id}>
                  <td><Link to="/rollouts/$runId" params={{ runId: r.id }} className="mono">{r.id}</Link></td>
                  <td><StatusBadge status={r.status} /></td>
                  <td>{r.profile}</td>
                  <td className="num">{r.step_count}</td>
                  <td className="num">{fmtCompact(r.tokens)}</td>
                  <td className="num">{fmtReward(r.terminal_reward)}</td>
                  <td className="muted">{fmtAgo(r.updated_at)}</td>
                </tr>
              ))}
              {recent.length === 0 ? <tr><td colSpan={7} className="empty">no rollouts yet</td></tr> : null}
            </tbody>
          </table>
        </Panel>
        <Flex direction="column" gap="3">
          <Panel title="Launch rollout">
            <LaunchForm />
          </Panel>
          <Panel title="Fleet" right={<Link to="/fleet">details →</Link>} flush>
            <table className="data-table">
              <tbody>
                {dash.data.nodes.map((n) => (
                  <tr key={n.id}>
                    <td className="mono">{n.id}</td>
                    <td className="muted">{n.role}</td>
                    <td><StatusBadge status={n.status} /></td>
                  </tr>
                ))}
                {dash.data.nodes.length === 0 ? <tr><td className="empty">no heartbeats yet</td></tr> : null}
              </tbody>
            </table>
          </Panel>
        </Flex>
      </Grid>
    </>
  );
}
