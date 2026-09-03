"use client";
import { useEffect, useState } from "react";
import { api, DEPTHS, type Meta, type Readiness, type TargetField } from "@/lib/api";
import { cropLon, REGIONS, type Region } from "@/lib/field";
import HeatmapCanvas from "@/components/HeatmapCanvas";
import ColorBar from "@/components/ColorBar";
import ProfileChart from "@/components/ProfileChart";
import AblationChart from "@/components/AblationChart";
import MetricsTable from "@/components/MetricsTable";
import EmbeddingCanvas from "@/components/EmbeddingCanvas";

type Tab = "recon" | "surface" | "skill" | "embed";
type Picked = { lat: number; lon: number } | null;

function useAsync<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [s, set] = useState<{ data?: T; err?: string; loading: boolean }>({ loading: true });
  useEffect(() => {
    let live = true;
    set({ loading: true });
    fn().then((d) => live && set({ data: d, loading: false }))
        .catch((e) => live && set({ err: e?.message ?? String(e), loading: false }));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return s;
}

export default function Page() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [fatal, setFatal] = useState<string | null>(null);
  const [ready, setReady] = useState<Readiness | null>(null);
  const [dateIdx, setDateIdx] = useState(0);
  const [model, setModel] = useState("");
  const [depth, setDepth] = useState(100);
  const [region, setRegion] = useState<Region>("full");
  const [tab, setTab] = useState<Tab>("recon");
  const [picked, setPicked] = useState<Picked>(null);
  const [compare, setCompare] = useState(false);

  useEffect(() => {
    api.ready().then(setReady);
    api.meta().then((m) => {
      setMeta(m);
      setModel(m.models.find((x) => x.is_default)?.key ?? m.models[0]?.key ?? "");
      setDepth(m.depths_m.includes(100) ? 100 : m.depths_m[Math.floor(m.depths_m.length / 2)]);
      setDateIdx(Math.max(0, m.dates.length - 1));
    }).catch((e) => setFatal(e?.message ?? String(e)));
  }, []);

  if (fatal) return <Fatal msg={fatal} ready={ready} />;
  if (!meta) return <Splash />;

  const date = meta.dates[dateIdx];
  const activeModel = meta.models.find((m) => m.key === model);

  return (
    <main>
      <header className="topbar">
        <div className="brand">
          <div className="logo">🌊</div>
          <div>
            <h1>OceanEmbed</h1>
            <p className="sub">Satellite surface → 0–1000 m subsurface temperature · {meta.region.name}</p>
          </div>
        </div>
        <div className="headStats">
          <Stat label="Best vs Argo" value={activeModel?.argo_rmse != null ? `${activeModel.argo_rmse.toFixed(3)} °C` : "—"} />
          <Stat label="GLORYS ceiling" value="0.728 °C" />
          <ReadyPill ready={ready} />
        </div>
      </header>

      <section className="controls">
        <div className="ctrlDate">
          <div className="ctrlRow">
            <button className="btn sq" onClick={() => setDateIdx((i) => Math.max(0, i - 1))}>‹</button>
            <input type="range" min={0} max={meta.dates.length - 1} value={dateIdx}
              onChange={(e) => setDateIdx(Number(e.target.value))} />
            <button className="btn sq" onClick={() => setDateIdx((i) => Math.min(meta.dates.length - 1, i + 1))}>›</button>
          </div>
          <div className="dateLbl"><b>{date}</b> <span className="muted">({dateIdx + 1}/{meta.dates.length})</span></div>
        </div>

        <label className="ctrlField">
          <span>Model</span>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {meta.models.map((m) => (
              <option key={m.key} value={m.key}>
                {m.label}{m.argo_rmse != null ? ` — ${m.argo_rmse.toFixed(3)}°C` : ""}
              </option>
            ))}
          </select>
        </label>

        <div className="ctrlField">
          <span>Region</span>
          <div className="seg">
            {REGIONS.map((r) => (
              <button key={r.key} className={region === r.key ? "on" : ""} onClick={() => setRegion(r.key)}>{r.label}</button>
            ))}
          </div>
        </div>

        <div className="tabs">
          {([["recon", "Reconstruction"], ["surface", "Surface inputs"], ["skill", "Skill"], ["embed", "Embedding"]] as [Tab, string][])
            .map(([k, l]) => <button key={k} className={tab === k ? "tab on" : "tab"} onClick={() => setTab(k)}>{l}</button>)}
        </div>
      </section>

      {tab === "recon" && (
        <>
          <DepthChips depths={meta.depths_m} depth={depth} onDepth={setDepth}
            compare={compare} onCompare={setCompare} />
          <ReconstructionView date={date} model={model} depth={depth} region={region}
            compare={compare} picked={picked} onPick={(lat, lon) => setPicked({ lat, lon })} />
        </>
      )}
      {tab === "surface" && <SurfaceView date={date} region={region} channels={meta.channels} />}
      {tab === "skill" && <SkillView model={model} />}
      {tab === "embed" && <EmbeddingView date={date} model={model} />}
    </main>
  );
}

/* ---------------------------------------------------------------- reconstruction */
function ReconstructionView({ date, model, depth, region, compare, picked, onPick }: {
  date: string; model: string; depth: number; region: Region; compare: boolean;
  picked: Picked; onPick: (lat: number, lon: number) => void;
}) {
  const recon = useAsync(() => api.reconstruction(date, depth, model), [date, depth, model]);
  const tgt = useAsync<TargetField | null>(() => (compare ? api.target(date, depth) : Promise.resolve(null)),
    [date, depth, compare]);

  const scale = recon.data ? { vmin: recon.data.vmin, vmax: recon.data.vmax } : undefined;

  return (
    <section className="grid recon">
      <div className="panel mapPanel">
        <div className="panelHead">
          <h2>Reconstruction @ {depth} m</h2>
          <span className="muted">{date} · {model}</span>
        </div>
        {recon.loading && <Skeleton h={360} />}
        {recon.err && <ErrBox msg={recon.err} />}
        {recon.data && (
          <div className={compare && tgt.data ? "compare" : ""}>
            <figure>
              {compare && tgt.data && <figcaption>Model</figcaption>}
              <HeatmapCanvas field={cropLon(recon.data, region)} picked={picked} onPick={onPick} />
            </figure>
            {compare && tgt.data && (
              <figure>
                <figcaption>GLORYS (target)</figcaption>
                <HeatmapCanvas field={cropLon(tgt.data, region)} vmin={scale?.vmin} vmax={scale?.vmax}
                  picked={picked} onPick={onPick} />
              </figure>
            )}
          </div>
        )}
        {recon.data && <ColorBar colormap={recon.data.colormap} vmin={recon.data.vmin} vmax={recon.data.vmax} units="°C" />}
        <p className="muted small">Click anywhere on the map to extract the full depth profile at that point.</p>
      </div>

      <div className="panel profilePanel">
        <div className="panelHead"><h2>Depth profile</h2></div>
        {!picked && <div className="empty">Click the map to pick a location.</div>}
        {picked && <ProfilePanel date={date} model={model} picked={picked} />}
      </div>
    </section>
  );
}

function ProfilePanel({ date, model, picked }: { date: string; model: string; picked: { lat: number; lon: number } }) {
  const prof = useAsync(() => api.profile(date, picked.lat, picked.lon, model), [date, model, picked.lat, picked.lon]);
  return (
    <>
      <div className="muted small">{picked.lat.toFixed(2)}°N, {picked.lon.toFixed(2)}°E</div>
      {prof.loading && <Skeleton h={360} />}
      {prof.err && <ErrBox msg={prof.err} />}
      {prof.data && <ProfileChart p={prof.data} />}
    </>
  );
}

function DepthChips({ depths, depth, onDepth, compare, onCompare }: {
  depths: number[]; depth: number; onDepth: (d: number) => void;
  compare: boolean; onCompare: (b: boolean) => void;
}) {
  return (
    <div className="depthBar">
      <span className="lbl">Depth</span>
      <div className="chips">
        {depths.map((d) => <button key={d} className={d === depth ? "chip on" : "chip"} onClick={() => onDepth(d)}>{d}</button>)}
      </div>
      <label className="toggle">
        <input type="checkbox" checked={compare} onChange={(e) => onCompare(e.target.checked)} /> GLORYS side-by-side
      </label>
    </div>
  );
}

/* ---------------------------------------------------------------- surface inputs */
function SurfaceView({ date, region, channels }: { date: string; region: Region; channels: Meta["channels"] }) {
  const s = useAsync(() => api.surfaceAll(date), [date]);
  return (
    <section className="panel">
      <div className="panelHead"><h2>Surface input fields</h2><span className="muted">{date} · 7 satellite-observable channels</span></div>
      {s.loading && <Skeleton h={300} />}
      {s.err && <ErrBox msg={s.err} />}
      {s.data && (
        <div className="surfaceGrid">
          {channels.map((c) => {
            const f = s.data![c.key];
            if (!f) return null;
            return (
              <div key={c.key} className="surfCard">
                <div className="surfHead"><b>{c.long_name}</b><span className="muted">{c.units}</span></div>
                <HeatmapCanvas field={cropLon(f, region)} />
                <ColorBar colormap={f.colormap} vmin={f.vmin} vmax={f.vmax} units={c.units} />
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

/* ---------------------------------------------------------------- skill */
function SkillView({ model }: { model: string }) {
  const metrics = useAsync(() => api.metrics(model), [model]);
  const abl = useAsync(() => api.ablation(), []);
  return (
    <section className="grid skill">
      <div className="panel">
        <div className="panelHead"><h2>Skill vs Argo — {model}</h2></div>
        {metrics.loading && <Skeleton h={360} />}
        {metrics.err && <ErrBox msg={metrics.err} />}
        {metrics.data && <MetricsTable data={metrics.data} />}
      </div>
      <div className="panel">
        <div className="panelHead"><h2>RMSE vs depth — ablation</h2><span className="muted">M0 → M4 + GLORYS ceiling</span></div>
        {abl.loading && <Skeleton h={360} />}
        {abl.err && <ErrBox msg={abl.err} />}
        {abl.data && <AblationChart data={abl.data} />}
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- embedding */
function EmbeddingView({ date, model }: { date: string; model: string }) {
  const emb = useAsync(() => api.embedding(date, model), [date, model]);
  return (
    <section className="panel">
      <div className="panelHead"><h2>OceanEmbed latent</h2><span className="muted">{date} · {model}</span></div>
      {emb.loading && <Skeleton h={280} />}
      {emb.err && <ErrBox msg={emb.err} />}
      {emb.data && <EmbeddingCanvas emb={emb.data} />}
    </section>
  );
}

/* ---------------------------------------------------------------- bits */
function Stat({ label, value }: { label: string; value: string }) {
  return <div className="stat"><span className="statLbl">{label}</span><span className="statVal">{value}</span></div>;
}
function ReadyPill({ ready }: { ready: Readiness | null }) {
  if (!ready) return <span className="pill amber">connecting…</span>;
  if (ready.ready) return <span className="pill green" title={ready.models.join(", ")}>API ready · {ready.models.length} model(s)</span>;
  const err = Object.entries(ready.errors).map(([k, v]) => `${k}: ${v}`).join("\n");
  return <span className="pill amber" title={err}>API up · not ready</span>;
}
function Skeleton({ h }: { h: number }) { return <div className="skeleton" style={{ height: h }} />; }
function ErrBox({ msg }: { msg: string }) { return <div className="errBox">⚠ {msg}</div>; }
function Splash() { return <div className="splash"><div className="logo big">🌊</div><p>Loading OceanEmbed…</p></div>; }
function Fatal({ msg, ready }: { msg: string; ready: Readiness | null }) {
  return (
    <div className="splash">
      <div className="logo big">🌊</div>
      <h2>Can’t reach the OceanEmbed API</h2>
      <p className="muted">{msg}</p>
      {ready && !ready.ready && (
        <pre className="errPre">{Object.entries(ready.errors).map(([k, v]) => `${k}: ${v}`).join("\n")}</pre>
      )}
      <p className="muted small">Start it with <code>cd server && ./run.sh</code> (needs the checkpoint + processed Zarr, or fake artifacts).</p>
    </div>
  );
}
