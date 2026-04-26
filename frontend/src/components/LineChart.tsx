import { useMemo } from "react";

export interface ChartSeries {
  name: string;
  color: string;
  points: { x: number; y: number }[];
}

interface Props {
  series: ChartSeries[];
  width?: number;
  height?: number;
  xLabel?: string;
  yLabel?: string;
  yDomain?: [number, number];
}

const PADDING = { top: 12, right: 16, bottom: 28, left: 44 };

export function LineChart({
  series,
  width = 480,
  height = 220,
  xLabel,
  yLabel,
  yDomain,
}: Props) {
  const flattened = useMemo(
    () => series.flatMap((s) => s.points),
    [series],
  );

  if (flattened.length === 0) {
    return (
      <div className="chart-empty" style={{ width, height }}>
        no data yet
      </div>
    );
  }

  const xs = flattened.map((p) => p.x);
  const ys = flattened.map((p) => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs, xMin + 1); // avoid div0 on a single point
  const [yMin, yMax] = yDomain ?? autoDomain(ys);

  const innerW = width - PADDING.left - PADDING.right;
  const innerH = height - PADDING.top - PADDING.bottom;

  const xScale = (x: number) =>
    PADDING.left + ((x - xMin) / (xMax - xMin)) * innerW;
  const yScale = (y: number) =>
    PADDING.top + innerH - ((y - yMin) / (yMax - yMin || 1)) * innerH;

  const xTicks = niceTicks(xMin, xMax, 5);
  const yTicks = niceTicks(yMin, yMax, 4);

  return (
    <svg
      role="img"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className="line-chart"
    >
      {/* Y grid + ticks */}
      {yTicks.map((t) => (
        <g key={`y-${t}`}>
          <line
            x1={PADDING.left}
            x2={width - PADDING.right}
            y1={yScale(t)}
            y2={yScale(t)}
            className="chart-grid"
          />
          <text
            x={PADDING.left - 6}
            y={yScale(t)}
            dominantBaseline="middle"
            textAnchor="end"
            className="chart-tick"
          >
            {formatNumber(t)}
          </text>
        </g>
      ))}

      {/* X ticks */}
      {xTicks.map((t) => (
        <g key={`x-${t}`}>
          <line
            x1={xScale(t)}
            x2={xScale(t)}
            y1={height - PADDING.bottom}
            y2={height - PADDING.bottom + 4}
            className="chart-grid"
          />
          <text
            x={xScale(t)}
            y={height - PADDING.bottom + 16}
            textAnchor="middle"
            className="chart-tick"
          >
            {formatNumber(t)}
          </text>
        </g>
      ))}

      {/* Series */}
      {series.map((s) => {
        if (s.points.length === 0) return null;
        const d = s.points
          .map(
            (p, i) =>
              `${i === 0 ? "M" : "L"} ${xScale(p.x).toFixed(1)} ${yScale(p.y).toFixed(1)}`,
          )
          .join(" ");
        return (
          <g key={s.name}>
            <path d={d} fill="none" stroke={s.color} strokeWidth={1.6} />
            {s.points.map((p, i) => (
              <circle
                key={`${s.name}-${i}`}
                cx={xScale(p.x)}
                cy={yScale(p.y)}
                r={2.4}
                fill={s.color}
              />
            ))}
          </g>
        );
      })}

      {/* Axis labels */}
      {xLabel ? (
        <text
          x={width / 2}
          y={height - 4}
          textAnchor="middle"
          className="chart-axis-label"
        >
          {xLabel}
        </text>
      ) : null}
      {yLabel ? (
        <text
          x={12}
          y={height / 2}
          textAnchor="middle"
          transform={`rotate(-90 12 ${height / 2})`}
          className="chart-axis-label"
        >
          {yLabel}
        </text>
      ) : null}

      {/* Legend */}
      <g transform={`translate(${PADDING.left}, ${PADDING.top - 4})`}>
        {series.map((s, i) => (
          <g key={s.name} transform={`translate(${i * 100}, 0)`}>
            <rect width={10} height={10} fill={s.color} y={-9} />
            <text x={14} y={0} className="chart-legend">
              {s.name}
            </text>
          </g>
        ))}
      </g>
    </svg>
  );
}

function autoDomain(ys: number[]): [number, number] {
  const min = Math.min(...ys);
  const max = Math.max(...ys);
  if (min === max) {
    const pad = Math.max(0.5, Math.abs(min) * 0.1);
    return [min - pad, max + pad];
  }
  const pad = (max - min) * 0.08;
  return [min - pad, max + pad];
}

function niceTicks(lo: number, hi: number, count: number): number[] {
  if (lo === hi) return [lo];
  const step = (hi - lo) / count;
  const ticks: number[] = [];
  for (let i = 0; i <= count; i++) ticks.push(lo + step * i);
  return ticks;
}

function formatNumber(n: number): string {
  if (Math.abs(n) >= 1000) return n.toFixed(0);
  if (Math.abs(n) >= 1) return n.toFixed(2);
  return n.toFixed(3);
}
