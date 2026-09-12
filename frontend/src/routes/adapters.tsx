import { Button, Text } from "@radix-ui/themes";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import type { ColumnDef } from "@tanstack/react-table";
import { useMemo, useState } from "react";
import { DataTable } from "../components/DataTable";
import { Panel } from "../components/Panel";
import { runEval } from "../lib/api";
import { fmtAgo, fmtReward } from "../lib/format";
import { useAdapters } from "../lib/query";
import type { AdapterRecord } from "../lib/types";

export function AdaptersPage() {
  const adapters = useAdapters();
  const qc = useQueryClient();
  const [busy, setBusy] = useState<string | null>(null);
  const evaluate = useMutation({
    mutationFn: (id: string) => runEval(id),
    onMutate: (id) => setBusy(id),
    onSettled: () => {
      setBusy(null);
      void qc.invalidateQueries({ queryKey: ["adapters"] });
    },
  });
  const columns = useMemo<ColumnDef<AdapterRecord, any>[]>(
    () => [
      { accessorKey: "id", header: "adapter", cell: (c) => <span className="mono">{c.getValue<string>()}</span> },
      { accessorKey: "base_model", header: "base model", cell: (c) => <span className="mono">{c.getValue<string>()}</span> },
      { accessorKey: "parent_id", header: "parent", cell: (c) => <span className="mono">{c.getValue<string | null>() ?? "–"}</span> },
      { accessorKey: "training_run_id", header: "training run", cell: (c) => c.getValue<string | null>() ? <Link to="/training/$trainingRunId" params={{ trainingRunId: c.getValue<string>() }} className="mono">{c.getValue<string>()}</Link> : "–" },
      { id: "tags", header: "tags", accessorFn: (r) => r.tags.join(", ") },
      { accessorKey: "eval_score", header: "eval", meta: { num: true }, cell: (c) => fmtReward(c.getValue<number | null>()) },
      { accessorKey: "created_at", header: "created", cell: (c) => <span className="muted">{fmtAgo(c.getValue<string>())}</span> },
      { id: "actions", header: "", cell: (c) => <Button size="1" variant="soft" disabled={busy !== null} onClick={() => evaluate.mutate(c.row.original.id)} data-testid={`eval-${c.row.original.id}`}>{busy === c.row.original.id ? "evaluating…" : "run eval"}</Button> },
    ],
    [busy, evaluate],
  );
  const data = useMemo(() => adapters.data ?? [], [adapters.data]);
  return (
    <>
      <Text size="4" weight="bold">Adapters</Text>
      {evaluate.error ? <Text size="1" color="red">{String(evaluate.error)}</Text> : null}
      <Panel title={`${data.length} adapters`} right="eval runs the built-in task set through the same rollout path" flush>
        <DataTable data={data} columns={columns} initialSort={[{ id: "created_at", desc: true }]} empty="no adapters yet; train one from completed rollouts" />
      </Panel>
    </>
  );
}
