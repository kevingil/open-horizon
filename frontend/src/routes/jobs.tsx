import { Text } from "@radix-ui/themes";
import { Link } from "@tanstack/react-router";
import type { ColumnDef } from "@tanstack/react-table";
import { useMemo } from "react";
import { DataTable } from "../components/DataTable";
import { Panel } from "../components/Panel";
import { StatusBadge } from "../components/StatusBadge";
import { fmtAgo, fmtTime } from "../lib/format";
import { useJobs } from "../lib/query";
import type { JobRecord } from "../lib/types";

const columns: ColumnDef<JobRecord, any>[] = [
  { accessorKey: "id", header: "job", cell: (c) => (c.row.original.kind === "rollout" ? <Link to="/rollouts/$runId" params={{ runId: c.getValue<string>() }} className="mono">{c.getValue<string>()}</Link> : <Link to="/training/$trainingRunId" params={{ trainingRunId: c.getValue<string>() }} className="mono">{c.getValue<string>()}</Link>) },
  { accessorKey: "kind", header: "kind" },
  { accessorKey: "status", header: "status", cell: (c) => <StatusBadge status={c.getValue<string>()} /> },
  { accessorKey: "attempts", header: "attempts", meta: { num: true } },
  { accessorKey: "lease_owner", header: "lease owner", cell: (c) => <span className="mono">{c.getValue<string | null>() ?? "–"}</span> },
  { accessorKey: "lease_until", header: "lease until", cell: (c) => <span className="muted">{c.getValue<string | null>() ? fmtTime(c.getValue<string>()) : "–"}</span> },
  { accessorKey: "cancel_requested", header: "cancel", cell: (c) => (c.getValue<boolean>() ? "requested" : "") },
  { accessorKey: "error", header: "error", cell: (c) => <span style={{ color: "var(--status-critical)" }}>{c.getValue<string | null>() ?? ""}</span> },
  { accessorKey: "created_at", header: "created", cell: (c) => <span className="muted">{fmtTime(c.getValue<string>())}</span> },
  { accessorKey: "updated_at", header: "updated", cell: (c) => <span className="muted">{fmtAgo(c.getValue<string>())}</span> },
];

export function JobsPage() {
  const jobs = useJobs();
  const data = useMemo(() => jobs.data ?? [], [jobs.data]);
  return (
    <>
      <Text size="4" weight="bold">Jobs</Text>
      <Panel title={`${data.length} jobs`} right="leased work; an expired lease is picked up by the next worker" flush>
        <DataTable data={data} columns={columns} initialSort={[{ id: "created_at", desc: true }]} empty="no jobs yet" />
      </Panel>
    </>
  );
}
