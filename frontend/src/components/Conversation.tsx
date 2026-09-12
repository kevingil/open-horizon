/**
 * Trajectory as a conversation: each policy action is a tool call card,
 * each environment step is the tool's output. Raw JSON stays one click away.
 */
import { Badge, Code } from "@radix-ui/themes";
import { useState } from "react";
import { fmtMs, fmtTime, parseJson } from "../lib/format";
import type { TrajectoryStep } from "../lib/types";

function summarize(step: TrajectoryStep): { tool: string | null; body: string } {
  const p = parseJson(step.content);
  if (!p) return { tool: null, body: step.content };
  const tool = typeof p.tool === "string" ? p.tool : null;
  if (step.actor === "policy") {
    const input = p.input as Record<string, unknown> | undefined;
    if (tool === "finish") return { tool, body: String(input?.summary ?? "") };
    if (tool === "run_command") return { tool, body: `$ ${String(input?.command ?? "")}` };
    if (tool === "read_file" || tool === "list_files") return { tool, body: String(input?.path ?? ".") };
    if (tool === "search") return { tool, body: `/${String(input?.pattern ?? "")}/ in ${String(input?.path ?? ".")}` };
    if (tool === "write_file") return { tool, body: `${String(input?.path ?? "")}\n${String(input?.content ?? "")}` };
    return { tool, body: JSON.stringify(input ?? {}, null, 2) };
  }
  if (typeof p.error === "string") return { tool, body: `error: ${p.error}` };
  if (typeof p.content === "string") return { tool, body: p.content };
  if (Array.isArray(p.entries)) return { tool, body: (p.entries as string[]).join("\n") };
  if (Array.isArray(p.matches)) return { tool, body: (p.matches as string[]).join("\n") || "(no matches)" };
  if (typeof p.stdout === "string" || typeof p.stderr === "string") {
    const rc = p.returncode;
    return { tool, body: `${String(p.stdout ?? "")}${p.stderr ? `\n[stderr]\n${String(p.stderr)}` : ""}\n[exit ${String(rc)}]` };
  }
  if (typeof p.summary === "string") return { tool, body: p.summary };
  if (p._truncated) return { tool, body: `(truncated)\n${String(p.preview ?? "")}` };
  return { tool, body: JSON.stringify(p, null, 2) };
}

export function Conversation({ steps }: { steps: TrajectoryStep[] }) {
  const [raw, setRaw] = useState<Set<number>>(new Set());
  if (steps.length === 0) return <div className="empty">no steps yet</div>;
  return (
    <div className="convo">
      {steps.map((step) => {
        const { tool, body } = summarize(step);
        const showRaw = raw.has(step.index);
        const isPolicy = step.actor === "policy";
        return (
          <div className="turn" key={step.index}>
            <div className="turn-meta">
              #{step.index} {fmtTime(step.at)}
              {step.duration_ms !== null && step.duration_ms !== undefined ? <><br />{fmtMs(step.duration_ms)}</> : null}
            </div>
            <div className={`turn-card ${isPolicy ? "turn-policy" : "turn-env"}`}>
              <div className="turn-head">
                <strong>{isPolicy ? "policy" : "environment"}</strong>
                {tool ? <Code size="1" variant="ghost">{tool}</Code> : null}
                <Badge size="1" variant="outline" color="gray">{step.kind}</Badge>
                {step.has_training_metadata ? <Badge size="1" variant="soft" color="blue">tokens recorded</Badge> : null}
                <span style={{ marginLeft: "auto" }}>
                  <button type="button" className="rt-reset muted" style={{ fontSize: 11, cursor: "pointer" }} onClick={() => setRaw((s) => { const n = new Set(s); n.has(step.index) ? n.delete(step.index) : n.add(step.index); return n; })}>
                    {showRaw ? "pretty" : "raw"}
                  </button>
                </span>
              </div>
              <div className="turn-body">{showRaw ? step.content : body}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
