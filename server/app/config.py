"""Settings — pydantic-settings, stdlib only (no numpy) so the app imports even where the
scientific stack is absent. Defaults are the repo's frozen artifact locations."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]   # server/app/config.py -> repo root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OCEANEMBED_", env_file=".env", extra="ignore")

    zarr: Path = ROOT / "data/processed/nio_daily.zarr"
    argo: Path = ROOT / "data/interim/argo_nio.parquet"
    results: Path = ROOT / "results"
    checkpoints: Path = ROOT / "checkpoints"
    stats: Path = ROOT / "data/processed/norm_stats.json"

    default_model: str = "m4_convlstm"
    served_models: str = "m4_convlstm,m2_unet,m3_oceanembed"
    split: str = "test"
    cors_origins: str = "http://localhost:3000"
    device: str = "cpu"

    def served_list(self):
        return [s.strip() for s in self.served_models.split(",") if s.strip()]

    def cors_list(self):
        return [s.strip() for s in self.cors_origins.split(",") if s.strip()]
