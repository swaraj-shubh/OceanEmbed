"use client";
import { useEffect, useRef } from "react";
import { colorFn } from "@/lib/colormaps";
import type { Field2D } from "@/lib/api";

function nearest(arr: number[], v: number) {
  let k = 0, best = Infinity;
  for (let i = 0; i < arr.length; i++) { const d = Math.abs(arr[i] - v); if (d < best) { best = d; k = i; } }
  return k;
}

export default function HeatmapCanvas({
  field, vmin, vmax, picked, onPick,
}: {
  field: Field2D;
  vmin?: number;
  vmax?: number;
  picked?: { lat: number; lon: number } | null;
  onPick?: (lat: number, lon: number) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const lo = vmin ?? field.vmin;
  const hi = vmax ?? field.vmax;

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const { values, lat, lon } = field;
    const H = lat.length, W = lon.length;
    cv.width = W; cv.height = H;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(W, H);
    const cf = colorFn(field.colormap, lo, hi);
    for (let r = 0; r < H; r++) {
      const latRow = H - 1 - r; // flip: north at top
      for (let c = 0; c < W; c++) {
        const v = values[latRow]?.[c];
        const p = (r * W + c) * 4;
        if (v == null || !isFinite(v)) { img.data[p + 3] = 0; continue; }
        const [rr, gg, bb] = cf(v);
        img.data[p] = rr; img.data[p + 1] = gg; img.data[p + 2] = bb; img.data[p + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [field, lo, hi]);

  const marker = picked
    ? { x: nearest(field.lon, picked.lon) / (field.lon.length - 1),
        y: 1 - nearest(field.lat, picked.lat) / (field.lat.length - 1) }
    : null;

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!onPick) return;
    const cv = ref.current!;
    const rect = cv.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    const { lat, lon } = field;
    const j = Math.max(0, Math.min(lon.length - 1, Math.round(fx * (lon.length - 1))));
    const i = Math.max(0, Math.min(lat.length - 1, Math.round((1 - fy) * (lat.length - 1))));
    onPick(lat[i], lon[j]);
  }

  return (
    <div className="mapWrap">
      <canvas
        ref={ref}
        onClick={handleClick}
        className="mapCanvas"
        style={{ cursor: onPick ? "crosshair" : "default" }}
      />
      {marker && <div className="marker" style={{ left: `${marker.x * 100}%`, top: `${marker.y * 100}%` }} />}
    </div>
  );
}
