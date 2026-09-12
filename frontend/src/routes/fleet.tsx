import { Text } from "@radix-ui/themes";
import type { ColumnDef } from "@tanstack/react-table";
import { useMemo } from "react";
import { DataTable } from "../components/DataTable";
import { Panel } from "../components/Panel";
import { StatusBadge } from "../components/StatusBadge";
import { fmtAgo } from "../lib/format";
import { useFleet, useRuntimeConfig } from "../lib/query";
import type { NodeRecord } from "../lib/types";

const columns: ColumnDef<NodeRecord, any>[] = [
  { accessorKey: "role", header: "role" },
  { accessorKey: "id", header: "node", cell: (c) => <span className="mono">{c.getValue<string>()}</span> },
  { accessorKey: "status", header: "status", cell: (c) => <StatusBadge status={c.getValue<string>()} /> },
  { accessorKey: "detail", header: "detail" },
  { id: "meta", header: "meta", meta: { wrap: true }, cell: (c) => <span className="mono muted">{Object.entries(c.row.original.meta).filter(([k]) => k !== "ping").map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`).join("  ")}</span> },
  { accessorKey: "last_seen", header: "last seen", cell: (c) => <span className="muted">{fmtAgo(c.getValue<string>())}</span> },
];

export function FleetPage() {
  const fleet = useFleet();
  const config = useRuntimeConfig();
  const data = useMemo(() => fleet.data ?? [], [fleet.data]);
  const down = data.filter((n) => n.status === "down").length;
  return (
    <>
      <Text size="4" weight="bold">Fleet</Text>
      <Panel title={`${data.length} nodes`} right={down ? `${down} down` : "heartbeats every 5s, probes every 15s"} flush>
        <DataTable data={data} columns={columns} initialSort={[{ id: "role", desc: false }]} empty="no heartbeats yet" />
      </Panel>
      {config.data ? (
        <Panel title="Runtime configuration">
          <dl className="kv">
            <dt>worker</dt><dd>{config.data.worker_id}</dd>
            <dt>store</dt><dd>{config.data.store_backend}</dd>
            <dt>environment</dt><dd>{config.data.env_backend}</dd>
            <dt>sandbox</dt><dd>{config.data.sandbox}</dd>
            <dt>trainer</dt><dd>{config.data.trainer_backend}</dd>
            <dt>policy</dt><dd>{config.data.policy_name}</dd>
            <dt>profiles</dt><dd>{config.data.profile_names.join(", ")} (default {config.data.default_profile})</dd>
            <dt>max parallel rollouts</dt><dd>{config.data.max_parallel_rollouts}</dd>
          </dl>
        </Panel>
      ) : null}
    </>
  );
}
