"use client";
import type { ProfileResponse } from "@/lib/api";
import { DEPTHS } from "@/lib/api";

const REPORT = [0, 50, 100, 200, 500, 1000];

export default function ProfileChart({ p }: { p: ProfileResponse }) {
  const W = 340, H = 440, m = { l: 48, r: 16, t: 18, b: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const dmax = 1000;

  const argo = p.argo?.obs_on_depths ?? null;
  const all = [...p.predicted, ...p.target, ...(argo ?? [])].filter((v): v is number => v != null && isFinite(v));
  if (all.length === 0) return <div className="muted">No profile at this cell.</div>;
  const tmin = Math.min(...all), tmax = Math.max(...all);
  const pad = (tmax - tmin) * 0.08 || 1;
  const lo = tmin - pad, hi = tmax + pad;

  const x = (t: number) => m.l + ((t - lo) / (hi - lo)) * iw;
  const y = (d: number) => m.t + (Math.sqrt(d) / Math.sqrt(dmax)) * ih;

  const line = (vals: (number | null)[]) => {
    let d = "", pen = false;
    vals.forEach((v, i) => {
      if (v == null || !isFinite(v)) { pen = false; return; }
      d += `${pen ? "L" : "M"}${x(v).toFixed(1)},${y(DEPTHS[i]).toFixed(1)}`;
      pen = true;
    });
    return d;
  };

  const tempTicks = [lo, (lo + hi) / 2, hi].map((t) => (t <= lo + 0.01 ? tmin : t));
  const at100 = p.argo?.point_metrics.find((r) => r.depth_m === 100);

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="temperature profile">
        {/* depth gridlines + labels */}
        {REPORT.map((d) => (
          <g key={d}>
            <line x1={m.l} x2={W - m.r} y1={y(d)} y2={y(d)} className="grid" />
            <text x={m.l - 8} y={y(d) + 3} className="axLbl" textAnchor="end">{d}</text>
          </g>
        ))}
        {tempTicks.map((t, i) => (
          <text key={i} x={x(t)} y={H - m.b + 18} className="axLbl" textAnchor="middle">{t.toFixed(1)}</text>
        ))}
        <text x={m.l - 34} y={m.t + ih / 2} className="axTitle" transform={`rotate(-90 ${m.l - 34} ${m.t + ih / 2})`}>Depth (m)</text>
        <text x={m.l + iw / 2} y={H - 6} className="axTitle" textAnchor="middle">Temperature (°C)</text>

        <path d={line(p.target)} className="lnTarget" />
        <path d={line(p.predicted)} className="lnPred" />
        {argo && argo.map((v, i) => (v != null && isFinite(v)
          ? <circle key={i} cx={x(v)} cy={y(DEPTHS[i])} r={3.2} className="ptArgo" /> : null))}
      </svg>

      <div className="legend">
        <span><i className="sw lnPred" /> Predicted ({p.model})</span>
        <span><i className="sw lnTarget" /> GLORYS</span>
        {argo && <span><i className="sw ptArgo" /> Argo (held-out)</span>}
      </div>

      {p.argo && (
        <div className="argoStats">
          <div><b>Argo match</b> · {p.argo.distance_km.toFixed(0)} km · {p.argo.days_off}d off</div>
          {at100 && (
            <div className="statRow">
              <span>RMSE@100m <b>{fmt(at100.rmse)}</b></span>
              <span>Bias@100m <b>{fmt(at100.bias)}</b></span>
              <span>Corr@100m <b>{fmt(at100.corr)}</b></span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function fmt(v: number | null) { return v == null ? "—" : v.toFixed(2); }
