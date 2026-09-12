import { Code } from "@radix-ui/themes";
import type { EventEnvelope } from "../lib/events";
import { fmtTime } from "../lib/format";

function describe(e: EventEnvelope): string {
  switch (e.kind) {
    case "rollout.queued": return `queued · profile ${e.payload.manifest.profile} · horizon ${e.payload.manifest.horizon}`;
    case "rollout.started": return "started";
    case "step.recorded": return `step #${e.payload.step.index} ${e.payload.step.actor} ${e.payload.step.kind}${e.payload.step.duration_ms != null ? ` · ${e.payload.step.duration_ms} ms` : ""}`;
    case "progress.ticked": return `turn ${e.payload.turn + 1}${e.payload.tool ? ` · ${e.payload.tool}` : ""} · ${e.payload.tokens} tok · $${e.payload.cost_usd.toFixed(4)}`;
    case "reward.computed": return `reward ${e.payload.reward.terminal_reward.toFixed(3)} · ${e.payload.reward.rubric}`;
    case "rollout.completed": return `completed · ${e.payload.manifest.step_count} steps · ${e.payload.manifest.tokens} tok`;
    case "rollout.failed": return `failed · ${e.payload.error}`;
    case "rollout.cancelled": return "cancelled";
    case "budget.exceeded": return `budget exceeded · $${e.payload.spent_usd.toFixed(4)} of $${e.payload.cap_usd.toFixed(2)}`;
    case "node.updated": return `${e.payload.node.role} ${e.payload.node.id} → ${e.payload.node.status} · ${e.payload.node.detail}`;
    case "log.line": return `${e.payload.level} ${e.payload.logger}: ${e.payload.message}`;
    case "training.queued": return `training queued · ${e.payload.record.trainer}`;
    case "training.started": return "training started";
    case "training.metric": return `step ${e.payload.metric.step} · loss ${e.payload.metric.loss.toFixed(4)}${e.payload.metric.mean_reward != null ? ` · reward ${e.payload.metric.mean_reward.toFixed(3)}` : ""}`;
    case "training.completed": return `training completed · adapter ${e.payload.record.adapter_out ?? "?"}`;
    case "training.failed": return `training failed · ${e.payload.error}`;
    case "adapter.published": return `adapter published · ${e.payload.adapter.id}`;
    case "eval.completed": return `eval · mean ${e.payload.report.mean_reward.toFixed(3)} over ${e.payload.report.per_task.length} tasks`;
    default: return "";
  }
}

export function EventTimeline({ events, showSubject = false, scroll = false }: { events: EventEnvelope[]; showSubject?: boolean; scroll?: boolean }) {
  if (events.length === 0) return <div className="empty">no events</div>;
  return (
    <div className={scroll ? "timeline scroll" : "timeline"}>
      {events.map((e) => (
        <div className="tl-row" key={e.seq}>
          <span className="mono">{fmtTime(e.at)}</span>
          <span><Code size="1" variant="ghost">{e.kind}</Code></span>
          <span>
            {showSubject && e.subject ? <span className="mono" style={{ marginRight: 8 }}>{e.subject.id}</span> : null}
            {describe(e)}
          </span>
        </div>
      ))}
    </div>
  );
}
