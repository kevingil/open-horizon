import { Button, Flex, Select, Text, TextField } from "@radix-ui/themes";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import type { ColumnDef } from "@tanstack/react-table";
import { useMemo, useState } from "react";
import { DataTable } from "../components/DataTable";
import { Panel } from "../components/Panel";
import { StatusBadge } from "../components/StatusBadge";
import { createRun } from "../lib/api";
import { fmtCompact, fmtDuration, fmtReward, fmtTime, fmtUsd } from "../lib/format";
import { useRuns, useRuntimeConfig } from "../lib/query";
import type { RunManifest, RunStatus } from "../lib/types";

const STATUSES: RunStatus[] = ["queued", "running", "completed", "failed", "cancelled"];

export function LaunchForm() {
  const qc = useQueryClient();
  const config = useRuntimeConfig();
  const [prompt, setPrompt] = useState("Explore this repository and summarise what it does.");
  const [horizon, setHorizon] = useState("6");
  const [profile, setProfile] = useState<string>("");
  const [count, setCount] = useState("1");
  const launch = useMutation({
    mutationFn: async () => {
      const n = Math.max(1, Math.min(50, Number(count) || 1));
      for (let i = 0; i < n; i++) {
        await createRun({
          prompt: prompt.trim(),
          repo_snapshot: ".",
          infra_target: "local",
          horizon: Number(horizon) || null,
          success_criteria: [],
          policy_profile: profile || null,
        });
      }
    },
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["runs"] }),
  });
  return (
    <Flex direction="column" gap="2">
      <TextField.Root size="1" value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="Task prompt" />
      <Flex gap="2" align="center" wrap="wrap">
        <Text size="1" className="muted">horizon</Text>
        <TextField.Root size="1" style={{ width: 60 }} value={horizon} onChange={(e) => setHorizon(e.target.value)} />
        <Text size="1" className="muted">count</Text>
        <TextField.Root size="1" style={{ width: 60 }} value={count} onChange={(e) => setCount(e.target.value)} />
        <Text size="1" className="muted">profile</Text>
        <Select.Root size="1" value={profile || "__default"} onValueChange={(v) => setProfile(v === "__default" ? "" : v)}>
          <Select.Trigger style={{ minWidth: 110 }} />
          <Select.Content>
            <Select.Item value="__default">default ({config.data?.default_profile ?? "…"})</Select.Item>
            {(config.data?.profile_names ?? []).map((p) => <Select.Item key={p} value={p}>{p}</Select.Item>)}
          </Select.Content>
        </Select.Root>
        <Button size="1" onClick={() => launch.mutate()} disabled={launch.isPending || !prompt.trim()} data-testid="queue-rollout">
          {launch.isPending ? "Queuing…" : "Queue"}
        </Button>
      </Flex>
      {launch.error ? <Text size="1" color="red">{String(launch.error)}</Text> : null}
    </Flex>
  );
}

const columns: ColumnDef<RunManifest, any>[] = [
  { accessorKey: "id", header: "run", cell: (c) => <Link to="/rollouts/$runId" params={{ runId: c.getValue<string>() }} className="mono">{c.getValue<string>()}</Link> },
  { accessorKey: "status", header: "status", cell: (c) => <StatusBadge status={c.getValue<string>()} /> },
  { accessorKey: "profile", header: "profile" },
  { accessorKey: "model_id", header: "model", cell: (c) => <span className="mono">{c.getValue<string>()}</span> },
  { accessorKey: "infra_target", header: "target" },
  { accessorKey: "horizon", header: "horizon", meta: { num: true } },
  { accessorKey: "step_count", header: "steps", meta: { num: true } },
  { accessorKey: "tokens", header: "tokens", meta: { num: true }, cell: (c) => fmtCompact(c.getValue<number>()) },
  { accessorKey: "cost_usd", header: "cost", meta: { num: true }, cell: (c) => fmtUsd(c.getValue<number>()) },
  { accessorKey: "terminal_reward", header: "reward", meta: { num: true }, cell: (c) => fmtReward(c.getValue<number | null>()) },
  { id: "duration", header: "duration", meta: { num: true }, accessorFn: (r) => (r.started_at ? (Date.parse(r.finished_at ?? new Date().toISOString()) - Date.parse(r.started_at)) : -1), cell: (c) => fmtDuration(c.row.original.started_at, c.row.original.finished_at) },
  { accessorKey: "created_at", header: "created", cell: (c) => <span className="muted">{fmtTime(c.getValue<string>())}</span> },
  { accessorKey: "adapter_id", header: "adapter", cell: (c) => <span className="mono">{c.getValue<string | null>() ?? ""}</span> },
];

export function RolloutsPage() {
  const [status, setStatus] = useState<RunStatus | "">("");
  const [profile, setProfile] = useState("");
  const runs = useRuns(status || undefined, profile || undefined);
  const config = useRuntimeConfig();
  const data = useMemo(() => runs.data ?? [], [runs.data]);
  return (
    <>
      <Flex align="center" justify="between" wrap="wrap" gap="2">
        <Text size="4" weight="bold">Rollouts</Text>
        <Flex gap="2" align="center">
          <Select.Root size="1" value={status || "all"} onValueChange={(v) => setStatus(v === "all" ? "" : (v as RunStatus))}>
            <Select.Trigger />
            <Select.Content>
              <Select.Item value="all">all statuses</Select.Item>
              {STATUSES.map((s) => <Select.Item key={s} value={s}>{s}</Select.Item>)}
            </Select.Content>
          </Select.Root>
          <Select.Root size="1" value={profile || "all"} onValueChange={(v) => setProfile(v === "all" ? "" : v)}>
            <Select.Trigger />
            <Select.Content>
              <Select.Item value="all">all profiles</Select.Item>
              {(config.data?.profile_names ?? []).map((p) => <Select.Item key={p} value={p}>{p}</Select.Item>)}
            </Select.Content>
          </Select.Root>
        </Flex>
      </Flex>
      <Panel title="Launch">
        <LaunchForm />
      </Panel>
      <Panel title={`${data.length} runs`} right={runs.isFetching ? "refreshing…" : ""} flush>
        <DataTable data={data} columns={columns} initialSort={[{ id: "created_at", desc: true }]} empty="no rollouts match" />
      </Panel>
    </>
  );
}
