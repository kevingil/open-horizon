/**
 * uPlot wrapper. One y-axis per chart (never two scales); series colours
 * come from the fixed categorical slots; crosshair + legend are on by default.
 */
import { useEffect, useRef } from "react";
import uPlot from "uplot";

export interface Series {
  label: string;
  values: (number | null)[];
  slot?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8;
  /** Draw as filled steps (counts) instead of a line. */
  bars?: boolean;
}

interface Props {
  /** Unix seconds (or plain step numbers when `timeAxis` is false), one per point. */
  x: number[];
  timeAxis?: boolean;
  series: Series[];
  height?: number;
  yLabel?: string;
  yRange?: [number, number];
  format?: (v: number) => string;
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

export function TimeSeries({ x, series, height = 200, yLabel, yRange, format, timeAxis = true }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const grid = cssVar("--viz-grid");
    const text = cssVar("--viz-text");
    const fmt = format ?? ((v: number) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(Math.abs(v) < 10 && v !== 0 ? 2 : 0)));
    const opts: uPlot.Options = {
      width: el.clientWidth || 600,
      height,
      cursor: { drag: { x: false, y: false }, points: { size: 8 } },
      legend: { live: true },
      scales: { x: { time: timeAxis }, y: yRange ? { range: yRange } : { range: (_u, min, max) => [Math.min(0, min), max === min ? max + 1 : max * 1.1] } },
      axes: [
        { stroke: text, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid }, font: "11px sans-serif", size: 28, label: timeAxis ? undefined : "step", labelFont: "11px sans-serif" },
        { stroke: text, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid }, font: "11px sans-serif", size: 52, label: yLabel, labelFont: "11px sans-serif", values: (_u, vals) => vals.map(fmt) },
      ],
      series: [
        { label: "time" },
        ...series.map((s, i) => {
          const color = cssVar(`--series-${s.slot ?? ((i % 8) + 1)}`);
          return s.bars
            ? { label: s.label, stroke: color, fill: color + "66", width: 1, paths: uPlot.paths.bars!({ size: [0.7, 20], radius: 0.2 }), points: { show: false }, value: (_u: uPlot, v: number | null) => (v === null ? "–" : fmt(v)) }
            : { label: s.label, stroke: color, width: 2, points: { show: x.length < 40, size: 6 }, spanGaps: false, value: (_u: uPlot, v: number | null) => (v === null ? "–" : fmt(v)) };
        }),
      ],
    };
    const data: uPlot.AlignedData = [x, ...series.map((s) => s.values)];
    plot.current?.destroy();
    plot.current = new uPlot(opts, data, el);
    const ro = new ResizeObserver(() => plot.current?.setSize({ width: el.clientWidth, height }));
    ro.observe(el);
    return () => {
      ro.disconnect();
      plot.current?.destroy();
      plot.current = null;
    };
    // Re-create on data change: cheaper than diffing for these sizes.
  }, [x, series, height, yLabel, yRange, format, timeAxis]);

  if (x.length === 0) return <div className="empty">no data in window</div>;
  return <div className="chart" ref={ref} />;
}
