import { useEffect, useMemo, useState } from "react";
import { fetchAdapters, runEval } from "../lib/api";
import type { AdapterRecord } from "../lib/types";

interface Node {
  record: AdapterRecord;
  children: Node[];
}

export function AdapterListPage() {
  const [adapters, setAdapters] = useState<AdapterRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [evaluating, setEvaluating] = useState<string | null>(null);

  const load = async () => {
    try {
      setAdapters(await fetchAdapters());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const tree = useMemo(() => buildTree(adapters), [adapters]);

  const onEval = async (adapterId: string) => {
    setEvaluating(adapterId);
    try {
      await runEval(adapterId);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setEvaluating(null);
    }
  };

  if (error) return <section className="panel">Adapter list error: {error}</section>;

  return (
    <div className="grid">
      <section className="panel panel-wide">
        <div className="panel-header">
          <h2>Adapters</h2>
          <span>{adapters.length}</span>
        </div>
        {adapters.length === 0 ? (
          <p className="muted">
            No adapters yet. Train one from a set of rollouts via{" "}
            <code>POST /api/training-runs</code>, the Training page, or{" "}
            <code>rl-train</code>.
          </p>
        ) : (
          <ul className="adapter-tree">
            {tree.map((node) => (
              <AdapterNode
                key={node.record.id}
                node={node}
                onEval={onEval}
                busy={evaluating}
              />
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function AdapterNode({
  node,
  onEval,
  busy,
}: {
  node: Node;
  onEval: (id: string) => void;
  busy: string | null;
}) {
  const a = node.record;
  return (
    <li>
      <div className="adapter-card">
        <div className="adapter-head">
          <strong>{a.id}</strong>
          {a.tags.map((t) => (
            <span key={t} className="tool-chip">
              {t}
            </span>
          ))}
          {a.eval_score !== null ? (
            <span className="badge badge-completed">eval {a.eval_score.toFixed(3)}</span>
          ) : (
            <span className="badge badge-pending">no eval</span>
          )}
        </div>
        <p className="muted">
          base: <code>{a.base_model}</code>
          {a.parent_id ? (
            <>
              {" "}
              · parent: <code>{a.parent_id}</code>
            </>
          ) : null}
          {a.training_run_id ? (
            <>
              {" "}
              · training: <code>{a.training_run_id}</code>
            </>
          ) : null}
        </p>
        <button
          className="action-btn action-btn-neutral"
          onClick={() => onEval(a.id)}
          disabled={busy !== null}
        >
          {busy === a.id ? "Evaluating..." : "Run eval"}
        </button>
      </div>
      {node.children.length > 0 ? (
        <ul className="adapter-children">
          {node.children.map((child) => (
            <AdapterNode key={child.record.id} node={child} onEval={onEval} busy={busy} />
          ))}
        </ul>
      ) : null}
    </li>
  );
}

function buildTree(records: AdapterRecord[]): Node[] {
  const byId: Record<string, Node> = {};
  for (const r of records) byId[r.id] = { record: r, children: [] };
  const roots: Node[] = [];
  for (const r of records) {
    if (r.parent_id && byId[r.parent_id]) {
      byId[r.parent_id].children.push(byId[r.id]);
    } else {
      roots.push(byId[r.id]);
    }
  }
  return roots;
}
