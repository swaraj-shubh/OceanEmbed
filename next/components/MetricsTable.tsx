"use client";
import type { Metrics } from "@/lib/api";

const REPORT = new Set([0, 50, 100, 200, 500, 1000]);
const fmt = (v: number | null, d = 3) => (v == null ? "—" : v.toFixed(d));

export default function MetricsTable({ data }: { data: Metrics }) {
  return (
    <div className="tableWrap">
      <table className="metrics">
        <thead>
          <tr><th>Depth (m)</th><th>n</th><th>RMSE</th><th>MAE</th><th>Bias</th><th>Corr</th><th>R²</th></tr>
        </thead>
        <tbody>
          {data.rows.map((r) => (
            <tr key={r.depth_m} className={REPORT.has(r.depth_m) ? "hi" : ""}>
              <td>{r.depth_m}</td>
              <td className="num dim">{r.n ?? "—"}</td>
              <td className="num">{fmt(r.rmse)}</td>
              <td className="num">{fmt(r.mae)}</td>
              <td className="num">{fmt(r.bias)}</td>
              <td className="num">{fmt(r.corr)}</td>
              <td className="num">{fmt(r.r2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="muted small">Highlighted rows are the six report depths. Source: <code>{data.source}</code></div>
    </div>
  );
}
