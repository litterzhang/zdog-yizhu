from pathlib import Path

import pytest

from yizhu_server.app import create_app
from yizhu_server.settings import Settings


@pytest.fixture()
def site(tmp_path: Path):
    root = tmp_path / "h5"
    root.mkdir()
    (root / "index.html").write_text("<html><body>Yizhu mock</body></html>")
    (root / "js").mkdir()
    (root / "js/app.a1b2c3d4.js").write_text("console.log('mock')")
    (root / "assets").mkdir()
    (root / "assets/leaf.png").write_bytes(b"test-image")
    return root


@pytest.fixture()
def client(site):
    app = create_app(Settings(h5_dist=site))
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health_is_explicitly_mock(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["mode"] == "mock"
    assert response.get_json()["framework"] == "flask"
    assert response.get_json()["auth"] == {"provider": "zdog-auth", "configured": False}
    assert response.get_json()["h5_ready"] is True
    assert response.headers["x-content-type-options"] == "nosniff"


def test_index_and_head(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Yizhu mock" in response.get_data(as_text=True)
    assert response.headers["cache-control"] == "no-cache"
    assert client.head("/").data == b""


def test_h5_browser_route_fallback(client):
    response = client.get("/pages/result/index?id=demo", headers={"Accept": "text/html"})
    assert response.status_code == 200
    assert "Yizhu mock" in response.get_data(as_text=True)


@pytest.mark.parametrize(
    "path", ["/api/missing", "/api", "/js/missing.js", "/assets/missing.png", "/missing"]
)
def test_missing_routes_are_not_html_success(client, path):
    assert client.get(path, headers={"Accept": "text/html"}).status_code == 404


def test_static_cache_policy(client):
    js = client.get("/js/app.a1b2c3d4.js")
    assert js.status_code == 200
    assert "immutable" in js.headers["cache-control"]
    image = client.get("/assets/leaf.png")
    assert image.status_code == 200
    assert "immutable" not in image.headers["cache-control"]


@pytest.mark.parametrize("name", [".env", "private.pem", "source.map", "settings.py"])
def test_private_file_types_are_not_served(client, site, name):
    (site / name).write_text("must-not-leak")
    assert client.get("/" + name).status_code == 404


def test_symlink_cannot_escape_dist(client, site):
    outside = site.parent / "private.json"
    outside.write_text('{"secret":"must-not-leak"}')
    (site / "leak.json").symlink_to(outside)
    assert client.get("/leak.json").status_code == 404


def test_encoded_traversal_is_rejected(client, site):
    (site.parent / "private.json").write_text('{"secret":"must-not-leak"}')
    assert client.get("/%2e%2e/private.json").status_code == 404


def test_index_symlink_cannot_escape_dist(client, site):
    outside = site.parent / "outside.html"
    outside.write_text("must-not-leak")
    (site / "index.html").unlink()
    (site / "index.html").symlink_to(outside)
    assert client.get("/").status_code == 404


def test_no_upload_or_business_mutation_endpoint(client):
    assert client.post("/api/plants", json={"ownerId": "untrusted"}).status_code == 405
    assert client.post("/api/upload", data=b"photo").status_code == 405


def test_untrusted_host_is_rejected(client):
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 400


def test_api_domain_is_trusted(client):
    response = client.get("/api/health", headers={"Host": "yizhu.942295.xyz"})
    assert response.status_code == 200
    assert response.get_json()["service"] == "yizhu-server"


def test_h5_domain_is_still_trusted(client):
    assert client.get("/", headers={"Host": "yizhuh5.942295.xyz"}).status_code == 200


def test_h5_can_read_api_across_domains(client):
    origin = "https://yizhuh5.942295.xyz"
    headers = {"Host": "yizhu.942295.xyz", "Origin": origin}
    response = client.get("/api/health", headers=headers)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert "access-control-allow-credentials" not in response.headers
    preflight = client.options(
        "/api/health",
        headers={**headers, "Access-Control-Request-Method": "GET"},
    )
    assert preflight.status_code == 204
    assert preflight.headers["access-control-allow-origin"] == origin


def test_untrusted_origin_gets_no_cors_permission(client):
    headers = {"Host": "yizhu.942295.xyz", "Origin": "https://evil.example"}
    response = client.get("/api/health", headers=headers)
    assert "access-control-allow-origin" not in response.headers
    preflight = client.options(
        "/api/health",
        headers={**headers, "Access-Control-Request-Method": "GET"},
    )
    assert preflight.status_code == 403


def test_missing_build_keeps_health_available(tmp_path):
    app = create_app(Settings(h5_dist=tmp_path / "not-built"))
    with app.test_client() as client:
        assert client.get("/").status_code == 503
        assert client.get("/api/health").get_json()["h5_ready"] is False
