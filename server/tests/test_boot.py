"""Wiring smoke test that needs NO scientific stack — boots the app with artifacts absent
and checks it degrades gracefully (health ok, ready 503, endpoints 503, docs served).
Runs anywhere fastapi+httpx exist:  python tests/test_boot.py  (or pytest)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_boot_degrades():
    with _client() as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/readyz").status_code == 503          # nothing loaded
        assert c.get("/openapi.json").status_code == 200    # routes registered
        assert c.get("/api/v1/meta").status_code == 503     # data endpoints refuse cleanly


if __name__ == "__main__":
    test_boot_degrades()
    print("test_boot OK")
