"""Authentication contract tests using synthetic credentials and mocked HTTP only."""

from dataclasses import replace

import pytest
import requests

from yizhu_server.app import create_app
from yizhu_server.authclient import AuthClient, AuthError
from yizhu_server.settings import Settings


@pytest.fixture(autouse=True)
def forbid_real_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Authentication tests must not contact a real service")

    monkeypatch.setattr(requests, "post", forbidden)


@pytest.fixture()
def settings(tmp_path):
    return Settings(
        h5_dist=tmp_path,
        auth_issuer="https://auth.example",
        auth_client_id="test-only-client",
        auth_client_secret="test-only-service-secret",
    )


@pytest.fixture()
def client(settings):
    app = create_app(settings)
    app.config["TESTING"] = True
    return app.test_client()


def upstream(monkeypatch, payload=None, status=200):
    calls = []

    class Response:
        status_code = status

        def json(self):
            if isinstance(payload, Exception):
                raise payload
            return payload

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(requests, "post", post)
    return calls


@pytest.mark.parametrize(
    "route,body,path",
    [
        (
            "/api/auth/login",
            {"username": "alice", "password": "test-password"},
            "/machine/password/login",
        ),
        (
            "/api/auth/register",
            {"username": "alice", "password": "test-password"},
            "/machine/register",
        ),
        ("/api/auth/wechat", {"code": "test-wechat-code"}, "/machine/wechat/miniprogram"),
    ],
)
def test_login_flows_delegate_and_only_return_the_token(
    client, settings, monkeypatch, route, body, path
):
    calls = upstream(
        monkeypatch,
        {
            "access_token": "test-user-token",
            "session_key": "not-for-the-browser",
            "client_secret": settings.auth_client_secret,
            "password_hash": "not-for-the-browser",
        },
    )
    response = client.post(route, json=body)
    assert response.status_code == 200
    assert response.get_json() == {"code": 200, "access_token": "test-user-token"}
    assert response.headers["Cache-Control"] == "no-store"
    url, options = calls[0]
    assert url == settings.auth_issuer + path
    assert options["json"]["client_id"] == settings.auth_client_id
    assert options["json"]["client_secret"] == settings.auth_client_secret
    assert options["allow_redirects"] is False
    assert options["timeout"] == (3, 8)
    for key, value in body.items():
        assert options["json"][key] == value
    assert settings.auth_client_secret not in response.get_data(as_text=True)


def test_missing_configuration_fails_closed_without_fake_login(tmp_path):
    client = create_app(Settings(h5_dist=tmp_path)).test_client()
    response = client.post(
        "/api/auth/login", json={"username": "alice", "password": "test-password"}
    )
    assert response.status_code == 503
    assert response.get_json()["error"] == "AUTH_NOT_CONFIGURED"
    assert "access_token" not in response.get_json()


@pytest.mark.parametrize(
    "body",
    [
        None,
        [],
        {},
        {"username": "alice"},
        {"username": "alice", "password": []},
        {"username": "alice", "password": "test-password", "ownerId": "someone-else"},
        {"username": "alice", "password": "test-password", "client_secret": "injected"},
    ],
)
def test_invalid_bodies_never_reach_auth_center(client, body):
    response = client.post("/api/auth/login", json=body)
    assert response.status_code == 400


def test_request_size_is_limited(client):
    response = client.post("/api/auth/login", json={"username": "alice", "password": "x" * 20000})
    assert response.status_code == 413


def test_identity_comes_only_from_verified_subject(client, monkeypatch):
    calls = upstream(
        monkeypatch, {"sub": "verified-user", "name": "观察者", "secret": "not-public"}
    )
    response = client.get(
        "/api/auth/me?ownerId=forged-user",
        headers={"Authorization": "Bearer test-user-token"},
    )
    assert response.status_code == 200
    assert response.get_json() == {
        "code": 200,
        "user": {"id": "verified-user", "nickname": "观察者"},
    }
    assert calls[0][0].endswith("/machine/userinfo")
    assert calls[0][1]["json"]["access_token"] == "test-user-token"


@pytest.mark.parametrize("header", ["", "test-user-token", "Basic test-user-token", "Bearer"])
def test_no_query_token_or_malformed_authorization(client, header):
    response = client.get("/api/auth/me?token=test-user-token", headers={"Authorization": header})
    assert response.status_code == 401


def test_missing_verified_subject_is_not_authenticated(client, monkeypatch):
    upstream(monkeypatch, {"name": "not-an-identity"})
    response = client.get("/api/auth/me", headers={"Authorization": "Bearer test-user-token"})
    assert response.status_code == 401


def test_auth_outage_is_not_treated_as_expired_login(client, monkeypatch):
    def timeout(*args, **kwargs):
        raise requests.Timeout("request-metadata-must-not-be-returned")

    monkeypatch.setattr(requests, "post", timeout)
    response = client.get("/api/auth/me", headers={"Authorization": "Bearer test-user-token"})
    assert response.status_code == 503
    assert response.get_json()["error"] == "AUTH_UNAVAILABLE"
    assert "request-metadata" not in response.get_data(as_text=True)


@pytest.mark.parametrize("status", [401, 403, 409, 429, 500])
def test_upstream_errors_are_sanitized(client, monkeypatch, status):
    upstream(monkeypatch, {"error_description": "private-upstream-information"}, status=status)
    response = client.post(
        "/api/auth/login", json={"username": "alice", "password": "test-password"}
    )
    assert response.status_code == (503 if status >= 500 else status)
    assert "private-upstream-information" not in response.get_data(as_text=True)


@pytest.mark.parametrize(
    "payload", [[], {}, {"access_token": ""}, {"access_token": "contains space"}]
)
def test_malformed_tokens_are_not_returned_as_success(client, monkeypatch, payload):
    upstream(monkeypatch, payload)
    response = client.post("/api/auth/wechat", json={"code": "test-wechat-code"})
    assert response.status_code == 502


def test_redirects_are_not_followed_with_credentials(client, monkeypatch):
    calls = upstream(monkeypatch, {}, status=302)
    response = client.post("/api/auth/wechat", json={"code": "test-wechat-code"})
    assert response.status_code == 502
    assert calls[0][1]["allow_redirects"] is False


def test_non_json_response_does_not_leak_content(client, monkeypatch):
    upstream(monkeypatch, ValueError("private-response-body"))
    response = client.post("/api/auth/wechat", json={"code": "test-wechat-code"})
    assert response.status_code == 502
    assert "private-response-body" not in response.get_data(as_text=True)


def test_binding_uses_the_current_token_not_a_client_user_id(client, monkeypatch):
    calls = upstream(monkeypatch, {"sub": "verified-user"})
    response = client.post(
        "/api/auth/bind/wechat",
        json={"code": "test-wechat-code"},
        headers={"Authorization": "Bearer test-user-token"},
    )
    assert response.status_code == 200
    assert len(calls) == 2
    assert calls[0][0].endswith("/machine/userinfo")
    assert calls[1][0].endswith("/machine/wechat/bind")
    assert calls[1][1]["json"]["access_token"] == "test-user-token"
    assert "user_id" not in calls[1][1]["json"]


def test_limiter_stops_requests_before_forwarding(settings, monkeypatch):
    client = create_app(replace(settings, auth_attempts_per_minute=2)).test_client()
    calls = upstream(monkeypatch, {}, status=401)
    for _ in range(2):
        assert (
            client.post(
                "/api/auth/login", json={"username": "alice", "password": "test-password"}
            ).status_code
            == 401
        )
    response = client.post(
        "/api/auth/login", json={"username": "alice", "password": "test-password"}
    )
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "60"
    assert len(calls) == 2


@pytest.mark.parametrize("origin", [
    "https://yizhuh5-h.942295.xyz",
    "https://yizhuh5.942295.xyz",
])
def test_h5_post_preflight_allows_authorization_but_not_arbitrary_origins(client, origin):
    response = client.options(
        "/api/auth/login",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type, Authorization",
        },
    )
    assert response.status_code == 204
    assert response.headers["Access-Control-Allow-Origin"] == origin
    assert "Access-Control-Allow-Credentials" not in response.headers
    response = client.post("/api/auth/login", headers={"Origin": "https://evil.example"}, json={})
    assert response.status_code == 403


def test_service_secret_is_not_in_settings_repr(settings):
    assert settings.auth_client_secret not in repr(settings)


def test_non_tls_remote_issuer_is_not_accepted(settings):
    auth = AuthClient(replace(settings, auth_issuer="http://evil.example"))
    with pytest.raises(AuthError) as error:
        auth.login("alice", "test-password")
    assert error.value.code == "AUTH_NOT_CONFIGURED"


def test_empty_optional_nickname_is_not_required(client, monkeypatch):
    calls = upstream(monkeypatch, {"access_token": "test-user-token"})
    response = client.post(
        "/api/auth/register",
        json={
            "username": "alice",
            "password": "test-password",
            "nickname": "  ",
        },
    )
    assert response.status_code == 200
    assert calls[0][1]["json"]["nickname"] is None


def test_application_error_with_http_200_is_not_success(client, monkeypatch):
    upstream(monkeypatch, {"error": "rejected", "error_description": "private-content"})
    response = client.post("/api/auth/wechat", json={"code": "test-code"})
    assert response.status_code == 502
    assert "private-content" not in response.get_data(as_text=True)


def test_cors_configuration_is_explicit(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert Settings.from_env().allowed_origins == (
        "https://yizhuh5-h.942295.xyz",
        "https://yizhuh5.942295.xyz",
    )
    monkeypatch.setenv("CORS_ORIGINS", "https://yizhuh5.942295.xyz, http://127.0.0.1:10086")
    assert Settings.from_env().allowed_origins == (
        "https://yizhuh5.942295.xyz",
        "http://127.0.0.1:10086",
    )
    monkeypatch.setenv("CORS_ORIGINS", "*")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_cli_uses_waitress_with_only_loopback_proxy_trust(monkeypatch):
    from yizhu_server import cli

    app = object()
    captured = {}
    monkeypatch.setattr("sys.argv", ["yizhu-server"])
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "create_app", lambda: app)

    def serve(application, **options):
        assert application is app
        captured.update(options)

    monkeypatch.setattr(cli, "serve", serve)
    cli.main()
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8091
    assert captured["trusted_proxy"] == "127.0.0.1"
    assert captured["trusted_proxy_count"] == 1
    assert captured["clear_untrusted_proxy_headers"] is True
