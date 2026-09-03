"use client";
import type { Ablation } from "@/lib/api";
import { SERIES_COLORS } from "@/lib/colormaps";

const REPORT = [0, 50, 100, 200, 500, 1000];

export default function AblationChart({ data }: { data: Ablation }) {
  const W = 380, H = 460, m = { l: 50, r: 16, t: 18, b: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const dmax = 1000;

  const rmses = data.series.flatMap((s) => s.rmse).filter((v): v is number => v != null && isFinite(v));
  const rmax = Math.max(1, ...rmses) * 1.05;

  const x = (r: number) => m.l + (r / rmax) * iw;
  const y = (d: number) => m.t + (Math.sqrt(d) / Math.sqrt(dmax)) * ih;

  const line = (s: { depths_m: number[]; rmse: (number | null)[] }) => {
    let d = "", pen = false;
    s.rmse.forEach((v, i) => {
      if (v == null || !isFinite(v)) { pen = false; return; }
      d += `${pen ? "L" : "M"}${x(v).toFixed(1)},${y(s.depths_m[i]).toFixed(1)}`; pen = true;
    });
    return d;
  };
  const rTicks = [0, rmax / 2, rmax];

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="RMSE vs depth">
        {REPORT.map((d) => (
          <g key={d}>
            <line x1={m.l} x2={W - m.r} y1={y(d)} y2={y(d)} className="grid" />
            <text x={m.l - 8} y={y(d) + 3} className="axLbl" textAnchor="end">{d}</text>
          </g>
        ))}
        {rTicks.map((r, i) => (
          <text key={i} x={x(r)} y={H - m.b + 18} className="axLbl" textAnchor="middle">{r.toFixed(2)}</text>
        ))}
        <text x={m.l - 36} y={m.t + ih / 2} className="axTitle" transform={`rotate(-90 ${m.l - 36} ${m.t + ih / 2})`}>Depth (m)</text>
        <text x={m.l + iw / 2} y={H - 6} className="axTitle" textAnchor="middle">RMSE vs Argo (°C)</text>

        {data.series.map((s, i) => {
          const ceiling = /glorys/i.test(s.label);
          return (
            <path key={s.label} d={line(s)}
              style={{ stroke: ceiling ? "#e8e896" : SERIES_COLORS[i % SERIES_COLORS.length] }}
              className={ceiling ? "ln ceiling" : "ln"} />
          );
        })}
      </svg>
      <div className="legend wrap">
        {data.series.map((s, i) => (
          <span key={s.label}>
            <i className="sw" style={{ background: /glorys/i.test(s.label) ? "#e8e896" : SERIES_COLORS[i % SERIES_COLORS.length] }} />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}
