"""End-to-end over every endpoint against the fake store. Run where torch/xarray exist:
    cd server && pytest
"""


def test_ready(client):
    r = client.get("/readyz").json()
    assert r["ready"] and set(r["models"]) == {"unet", "temporal"}


def test_meta(client):
    m = client.get("/api/v1/meta").json()
    assert len(m["depths_m"]) == 15 and len(m["channels"]) == 7
    assert {mm["key"] for mm in m["models"]} == {"unet", "temporal"}
    assert m["dates"], "no dates"


def test_surface(client):
    d = client.get("/api/v1/meta").json()["dates"][0]
    one = client.get(f"/api/v1/surface/{d}/sst").json()
    assert len(one["values"]) == 96 and len(one["values"][0]) == 176
    assert one["channel"] == "sst"
    assert client.get(f"/api/v1/surface/{d}/nope").status_code == 422
    assert len(client.get(f"/api/v1/surface/{d}").json()) == 7


def test_reconstruction_and_target(client):
    d = client.get("/api/v1/meta").json()["dates"][0]
    r = client.get("/api/v1/reconstruction", params={"date": d, "depth": 100}).json()
    assert r["depth_m"] == 100 and len(r["values"]) == 96
    t = client.get("/api/v1/target", params={"date": d, "depth": 100})
    assert t.status_code == 200
    assert client.get("/api/v1/reconstruction",
                      params={"date": d, "depth": 42}).status_code == 422   # bad depth


def test_profile_with_argo(client):
    r = client.get("/api/v1/profile",
                   params={"date": client.argo_date, "lat": 10.1, "lon": 80.2}).json()
    assert len(r["predicted"]) == 15
    assert r["argo"] is not None and r["argo"]["point_metrics"]        # matched the cast


def test_argo_nearby(client):
    r = client.get("/api/v1/argo",
                   params={"date": client.argo_date, "lat": 10.1, "lon": 80.2}).json()
    assert r["count"] >= 1


def test_metrics_and_ablation(client):
    rows = client.get("/api/v1/metrics", params={"model": "unet"}).json()["rows"]
    assert len(rows) == 15 and rows[0]["depth_m"] == 0
    series = client.get("/api/v1/metrics/ablation").json()["series"]
    assert any("GLORYS" in s["label"] for s in series)


def test_temporal_reconstruction_and_embedding(client):
    # window=7 needs history; the 7th val day is the earliest predictable one
    d = client.get("/api/v1/meta").json()["dates"][6]
    r = client.get("/api/v1/reconstruction", params={"date": d, "depth": 50, "model": "temporal"})
    assert r.status_code == 200
    e = client.get("/api/v1/embedding", params={"date": d, "model": "temporal"}).json()
    assert e["shape"] == [12, 22] and len(e["rgb"]) == 12 and len(e["explained_variance"]) == 3


def test_unknown_date_and_model(client):
    assert client.get("/api/v1/surface/1999-01-01/sst").status_code == 404
    assert client.get("/api/v1/reconstruction",
                      params={"date": "2022-01-01", "depth": 0, "model": "ghost"}
                      ).status_code == 404
