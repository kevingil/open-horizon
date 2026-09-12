/**
 * Dense sortable table on TanStack Table. Column defs decide alignment
 * through `meta.num`; rows are plain <tr> so the browser handles scale.
 */
import { flexRender, getCoreRowModel, getSortedRowModel, useReactTable, type ColumnDef, type SortingState } from "@tanstack/react-table";
import { useState } from "react";

declare module "@tanstack/react-table" {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  interface ColumnMeta<TData, TValue> {
    num?: boolean;
    wrap?: boolean;
  }
}

export function DataTable<T>({ data, columns, initialSort, empty }: { data: T[]; columns: ColumnDef<T, any>[]; initialSort?: SortingState; empty?: string }) {
  const [sorting, setSorting] = useState<SortingState>(initialSort ?? []);
  const table = useReactTable({ data, columns, state: { sorting }, onSortingChange: setSorting, getCoreRowModel: getCoreRowModel(), getSortedRowModel: getSortedRowModel() });
  if (data.length === 0) return <div className="empty">{empty ?? "nothing yet"}</div>;
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          {table.getHeaderGroups().map((hg) => (
            <tr key={hg.id}>
              {hg.headers.map((h) => (
                <th key={h.id} className={h.column.columnDef.meta?.num ? "num" : undefined} onClick={h.column.getToggleSortingHandler()}>
                  {flexRender(h.column.columnDef.header, h.getContext())}
                  {{ asc: " ▲", desc: " ▼" }[h.column.getIsSorted() as string] ?? ""}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id}>
              {row.getVisibleCells().map((cell) => (
                <td key={cell.id} className={[cell.column.columnDef.meta?.num ? "num" : "", cell.column.columnDef.meta?.wrap ? "wrap" : ""].join(" ").trim() || undefined}>
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
