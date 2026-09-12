import { Tooltip } from "@radix-ui/themes";
import type { RewardBin } from "../lib/types";

export function Histogram({ bins }: { bins: RewardBin[] }) {
  const max = Math.max(1, ...bins.map((b) => b.count));
  const total = bins.reduce((a, b) => a + b.count, 0);
  if (total === 0) return <div className="empty">no scored runs in window</div>;
  return (
    <div>
      <div className="hist" role="img" aria-label={`Reward histogram, ${total} runs`}>
        {bins.map((b) => (
          <Tooltip key={b.lo} content={`${b.lo.toFixed(2)} to ${b.hi.toFixed(2)}: ${b.count} run${b.count === 1 ? "" : "s"}`}>
            <div className="hist-bar" style={{ height: `${(b.count / max) * 100}%`, opacity: b.count === 0 ? 0.25 : 1 }} />
          </Tooltip>
        ))}
      </div>
      <div className="hist-axis"><span>-1.0</span><span>0</span><span>+1.0</span></div>
    </div>
  );
}
