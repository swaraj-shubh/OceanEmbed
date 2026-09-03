"""Typed responses — the contract the /next frontend mirrors. Fields are permissive
(Optional, null-friendly) because ocean fields carry NaN on land/missing, which serialises
to null. See README §8."""
from typing import Optional

from pydantic import BaseModel

Row = list[Optional[float]]     # a grid row: floats with null for land/missing


class Field2D(BaseModel):
    values: list[Row]           # [lat][lon], row-major; null = land/missing
    lat: list[float]
    lon: list[float]
    units: str
    vmin: float
    vmax: float
    colormap: str


class SurfaceField(Field2D):
    channel: str
    long_name: str


class ReconstructionField(Field2D):
    date: str
    depth_m: int
    model: str


class TargetField(Field2D):
    date: str
    depth_m: int


class ChannelMeta(BaseModel):
    key: str
    long_name: str
    units: str
    colormap: str


class ModelMeta(BaseModel):
    key: str
    label: str
    kind: str
    window: int
    is_default: bool
    argo_rmse: Optional[float] = None
    n_dates: int


class BBox(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class MetaResponse(BaseModel):
    region: dict
    grid: dict
    dates: list[str]
    depths_m: list[int]
    report_depths_m: list[int]
    channels: list[ChannelMeta]
    models: list[ModelMeta]


class PointMetric(BaseModel):
    depth_m: int
    rmse: Optional[float] = None
    mae: Optional[float] = None
    bias: Optional[float] = None
    corr: Optional[float] = None


class ArgoMatch(BaseModel):
    profile_id: str
    lat: float
    lon: float
    distance_km: float
    days_off: int
    obs_on_depths: Row
    point_metrics: list[PointMetric]


class ProfileResponse(BaseModel):
    cell: dict
    date: str
    model: str
    depths_m: list[int]
    predicted: Row
    target: Row
    argo: Optional[ArgoMatch] = None


class ArgoNearby(BaseModel):
    profile_id: str
    lat: float
    lon: float
    time: str
    distance_km: float


class ArgoListResponse(BaseModel):
    date: str
    count: int
    profiles: list[ArgoNearby]


class MetricRow(BaseModel):
    depth_m: int
    n: Optional[int] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    bias: Optional[float] = None
    corr: Optional[float] = None
    r2: Optional[float] = None


class MetricsResponse(BaseModel):
    model: str
    source: str
    rows: list[MetricRow]


class AblationSeries(BaseModel):
    label: str
    source: str
    depths_m: list[int]
    rmse: Row


class AblationResponse(BaseModel):
    series: list[AblationSeries]


class EmbeddingResponse(BaseModel):
    date: str
    model: str
    shape: list[int]                 # [h, w]
    rgb: list[list[list[float]]]     # [h][w][3], 0..1
    explained_variance: list[float]


class ReadyResponse(BaseModel):
    ready: bool
    components: dict
    models: list[str]
    errors: dict
