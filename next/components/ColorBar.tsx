"use client";
import { gradientCss } from "@/lib/colormaps";

export default function ColorBar({ colormap, vmin, vmax, units }:
  { colormap: string; vmin: number; vmax: number; units: string }) {
  const mid = (vmin + vmax) / 2;
  const f = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1));
  return (
    <div className="colorbar">
      <div className="colorbarBar" style={{ background: gradientCss(colormap) }} />
      <div className="colorbarTicks">
        <span>{f(vmin)}</span><span>{f(mid)}</span><span>{f(vmax)} {units}</span>
      </div>
    </div>
  );
}
