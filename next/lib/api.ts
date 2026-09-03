// Typed client mirroring server/app/schemas.py. One place the frontend and backend agree.
const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "/api/v1";

export type Cell = number | null; // null = land / missing
export type Row = Cell[];

export interface Field2D {
  values: Row[]; // [lat][lon]
  lat: number[];
  lon: number[];
  units: string;
  vmin: number;
  vmax: number;
  colormap: string;
}
export interface SurfaceField extends Field2D { channel: string; long_name: string; }
export interface ReconstructionField extends Field2D { date: string; depth_m: number; model: string; }
export interface TargetField extends Field2D { date: string; depth_m: number; }

export interface ChannelMeta { key: string; long_name: string; units: string; colormap: string; }
export interface ModelMeta {
  key: string; label: string; kind: string; window: number;
  is_default: boolean; argo_rmse: number | null; n_dates: number;
}
export interface Meta {
  region: { name: string; bbox: { lat_min: number; lat_max: number; lon_min: number; lon_max: number } };
  grid: { model_shape: number[]; res_deg: number; lat: number[]; lon: number[] };
  dates: string[];
  depths_m: number[];
  report_depths_m: number[];
  channels: ChannelMeta[];
  models: ModelMeta[];
}

export interface PointMetric { depth_m: number; rmse: Cell; mae: Cell; bias: Cell; corr: Cell; }
export interface ArgoMatch {
  profile_id: string; lat: number; lon: number; distance_km: number; days_off: number;
  obs_on_depths: Row; point_metrics: PointMetric[];
}
export interface ProfileResponse {
  cell: { lat: number; lon: number };
  date: string; model: string; depths_m: number[];
  predicted: Row; target: Row; argo: ArgoMatch | null;
}

export interface ArgoNearby { profile_id: string; lat: number; lon: number; time: string; distance_km: number; }
export interface ArgoList { date: string; count: number; profiles: ArgoNearby[]; }

export interface MetricRow {
  depth_m: number; n: number | null; rmse: Cell; mae: Cell; bias: Cell; corr: Cell; r2: Cell;
}
export interface Metrics { model: string; source: string; rows: MetricRow[]; }
export interface AblationSeries { label: string; source: string; depths_m: number[]; rmse: Row; }
export interface Ablation { series: AblationSeries[]; }
export interface Embedding { date: string; model: string; shape: number[]; rgb: number[][][]; explained_variance: number[]; }
export interface Readiness { ready: boolean; components: Record<string, boolean>; models: string[]; errors: Record<string, string>; }

async function get<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const q = params
    ? "?" + Object.entries(params).filter(([, v]) => v !== undefined && v !== "")
        .map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`).join("&")
    : "";
  const res = await fetch(`${BASE}${path}${q}`, { cache: "no-store" });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch {}
    throw new ApiError(res.status, detail);
  }
  return res.json();
}

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

export const api = {
  ready: () => fetch("/readyz", { cache: "no-store" }).then(async (r) => r.json() as Promise<Readiness>).catch(() => null),
  meta: () => get<Meta>("/meta"),
  surfaceAll: (date: string) => get<Record<string, SurfaceField>>(`/surface/${date}`),
  reconstruction: (date: string, depth: number, model?: string) =>
    get<ReconstructionField>("/reconstruction", { date, depth, model }),
  target: (date: string, depth: number) => get<TargetField>("/target", { date, depth }),
  profile: (date: string, lat: number, lon: number, model?: string) =>
    get<ProfileResponse>("/profile", { date, lat, lon, model }),
  argo: (date: string, lat: number, lon: number) => get<ArgoList>("/argo", { date, lat, lon }),
  metrics: (model: string) => get<Metrics>("/metrics", { model }),
  ablation: () => get<Ablation>("/metrics/ablation"),
  embedding: (date: string, model?: string) => get<Embedding>("/embedding", { date, model }),
};

export const DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000];
