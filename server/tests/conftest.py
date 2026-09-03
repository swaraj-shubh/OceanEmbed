"""Functional fixtures — need the scientific stack; skipped where it's absent (dev boxes)."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("xarray")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # /server on path
from scripts.make_fake_artifacts import ARGO_DATE, build              # noqa: E402


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    env = build(tmp_path_factory.mktemp("artifacts"))
    import os
    os.environ.update(env)
    from fastapi.testclient import TestClient
    from app.main import create_app
    with TestClient(create_app()) as c:
        c.argo_date = ARGO_DATE           # stash for tests
        yield c
