import { Button, Flex, Select, Text } from "@radix-ui/themes";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "@tanstack/react-router";
import type { ColumnDef } from "@tanstack/react-table";
import { useMemo, useState } from "react";
import { DataTable } from "../components/DataTable";
import { Panel } from "../components/Panel";
import { StatusBadge } from "../components/StatusBadge";
import { createTrainingRun } from "../lib/api";
import { fmtAgo, fmtReward, fmtTime } from "../lib/format";
import { useAdapters, useRuns, useTrainingRuns } from "../lib/query";
import type { TrainingRunRecord } from "../lib/types";

const columns: ColumnDef<TrainingRunRecord, any>[] = [
  { accessorKey: "id", header: "training run", cell: (c) => <Link to="/training/$trainingRunId" params={{ trainingRunId: c.getValue<string>() }} className="mono">{c.getValue<string>()}</Link> },
  { accessorKey: "status", header: "status", cell: (c) => <StatusBadge status={c.getValue<string>()} /> },
  { accessorKey: "trainer", header: "trainer" },
  { id: "samples", header: "samples", meta: { num: true }, accessorFn: (r) => r.sample_run_ids.length },
  { id: "steps", header: "steps", meta: { num: true }, accessorFn: (r) => r.metrics.length },
  { id: "loss", header: "final loss", meta: { num: true }, accessorFn: (r) => r.metrics.at(-1)?.loss ?? null, cell: (c) => (c.getValue<number | null>() == null ? "–" : c.getValue<number>().toFixed(4)) },
  { accessorKey: "adapter_in", header: "parent", cell: (c) => <span className="mono">{c.getValue<string | null>() ?? "–"}</span> },
  { accessorKey: "adapter_out", header: "child", cell: (c) => <span className="mono">{c.getValue<string | null>() ?? "–"}</span> },
  { accessorKey: "created_at", header: "created", cell: (c) => <span className="muted">{fmtTime(c.getValue<string>())}</span> },
  { accessorKey: "updated_at", header: "updated", cell: (c) => <span className="muted">{fmtAgo(c.getValue<string>())}</span> },
];

function TrainForm() {
  const runs = useRuns("completed");
  const adapters = useAdapters();
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [parent, setParent] = useState("");
  const [steps, setSteps] = useState("8");
  const start = useMutation({
    mutationFn: () => createTrainingRun({ sample_run_ids: Array.from(selected), parent_adapter_id: parent || null, hyperparams: { steps: Number(steps) || 8 } }),
    onSuccess: (r) => {
      void qc.invalidateQueries({ queryKey: ["training-runs"] });
      navigate({ to: "/training/$trainingRunId", params: { trainingRunId: r.training_run_id } });
    },
  });
  const candidates = (runs.data ?? []).slice(0, 40);
  const toggle = (id: string) => setSelected((s) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; });
  return (
    <Flex direction="column" gap="2">
      <Flex gap="2" align="center" wrap="wrap">
        <Text size="1" className="muted">{selected.size} of {candidates.length} completed rollouts selected</Text>
        <Button size="1" variant="ghost" onClick={() => setSelected(new Set(candidates.map((r) => r.id)))}>select all</Button>
        <Button size="1" variant="ghost" onClick={() => setSelected(new Set())}>clear</Button>
        <Text size="1" className="muted">parent</Text>
        <Select.Root size="1" value={parent || "__none"} onValueChange={(v) => setParent(v === "__none" ? "" : v)}>
          <Select.Trigger style={{ minWidth: 140 }} />
          <Select.Content>
            <Select.Item value="__none">base model</Select.Item>
            {(adapters.data ?? []).map((a) => <Select.Item key={a.id} value={a.id}>{a.id}</Select.Item>)}
          </Select.Content>
        </Select.Root>
        <Text size="1" className="muted">steps</Text>
        <Select.Root size="1" value={steps} onValueChange={setSteps}>
          <Select.Trigger />
          <Select.Content>{["4", "8", "16", "32"].map((s) => <Select.Item key={s} value={s}>{s}</Select.Item>)}</Select.Content>
        </Select.Root>
        <Button size="1" onClick={() => start.mutate()} disabled={selected.size === 0 || start.isPending} data-testid="start-training">
          {start.isPending ? "Starting…" : "Start training"}
        </Button>
      </Flex>
      <div className="table-scroll" style={{ maxHeight: 220 }}>
        <table className="data-table">
          <tbody>
            {candidates.map((r) => (
              <tr key={r.id} onClick={() => toggle(r.id)} style={{ cursor: "pointer" }} data-testid="sample-row">
                <td style={{ width: 24 }}><input type="checkbox" readOnly checked={selected.has(r.id)} /></td>
                <td className="mono">{r.id}</td>
                <td className="muted">{r.profile}</td>
                <td className="num">{r.step_count} steps</td>
                <td className="num">{fmtReward(r.terminal_reward)}</td>
                <td className="muted">{fmtAgo(r.finished_at)}</td>
              </tr>
            ))}
            {candidates.length === 0 ? <tr><td className="empty">no completed rollouts to train on</td></tr> : null}
          </tbody>
        </table>
      </div>
      {start.error ? <Text size="1" color="red">{String(start.error)}</Text> : null}
    </Flex>
  );
}

export function TrainingPage() {
  const runs = useTrainingRuns();
  const data = useMemo(() => runs.data ?? [], [runs.data]);
  return (
    <>
      <Text size="4" weight="bold">Training</Text>
      <Panel title="New training run">
        <TrainForm />
      </Panel>
      <Panel title={`${data.length} training runs`} flush>
        <DataTable data={data} columns={columns} initialSort={[{ id: "created_at", desc: true }]} empty="no training runs yet" />
      </Panel>
    </>
  );
}
