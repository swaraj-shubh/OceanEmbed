"""HTTP layer — thin: validate params, call one service, return typed schema. No numpy here."""
from fastapi import APIRouter, Depends, Query, Request

from app import schemas, services
from app.config import Settings
from app.state import AppState

router = APIRouter(prefix="/api/v1")


def get_state(request: Request) -> AppState:
    return request.app.state.oe


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


@router.get("/meta", response_model=schemas.MetaResponse)
def meta(state=Depends(get_state), settings=Depends(get_settings)):
    return services.meta(state, settings)


@router.get("/surface/{date}")
def surface_all(date: str, grid: str = Query("model", pattern="^(model|native)$"),
                state=Depends(get_state)):
    return services.surface_all(state, date, grid)


@router.get("/surface/{date}/{channel}", response_model=schemas.SurfaceField)
def surface_one(date: str, channel: str,
                grid: str = Query("model", pattern="^(model|native)$"),
                state=Depends(get_state)):
    return services.surface(state, date, channel, grid)


@router.get("/reconstruction", response_model=schemas.ReconstructionField)
def reconstruction(date: str, depth: int, model: str | None = None,
                   state=Depends(get_state)):
    return services.reconstruction(state, date, depth, model)


@router.get("/target", response_model=schemas.TargetField)
def target(date: str, depth: int, grid: str = Query("model", pattern="^(model|native)$"),
           state=Depends(get_state)):
    return services.target(state, date, depth, grid)


@router.get("/profile", response_model=schemas.ProfileResponse)
def profile(date: str, lat: float, lon: float, model: str | None = None,
            state=Depends(get_state)):
    return services.profile(state, date, lat, lon, model)


@router.get("/argo", response_model=schemas.ArgoListResponse)
def argo(date: str, lat: float, lon: float, radius_deg: float = 1.5, max_days: int = 3,
         state=Depends(get_state)):
    return services.argo_nearby(state, date, lat, lon, radius_deg, max_days)


@router.get("/metrics/ablation", response_model=schemas.AblationResponse)
def ablation(state=Depends(get_state)):
    return services.ablation(state)


@router.get("/metrics", response_model=schemas.MetricsResponse)
def metrics(model: str, state=Depends(get_state)):
    return services.metrics(state, model)


@router.get("/embedding", response_model=schemas.EmbeddingResponse)
def embedding(date: str, model: str | None = None, state=Depends(get_state)):
    return services.embedding(state, date, model)
