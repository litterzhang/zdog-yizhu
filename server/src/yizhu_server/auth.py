"""Small Flask authentication blueprint. Only the auth center determines identity."""

import time
from functools import wraps
from threading import Lock

from flask import Blueprint, current_app, g, request

from yizhu_server.authclient import AuthClient, AuthError


def client() -> AuthClient:
    return current_app.extensions["zdog_auth"]


class AttemptLimiter:
    """Bounded, thread-safe per-IP limiter for this single-process personal service."""

    def __init__(self, limit: int = 20):
        self.limit = limit
        self.buckets: dict[str, tuple[float, int]] = {}
        self.lock = Lock()

    def allow(self, address: str) -> bool:
        now = time.monotonic()
        with self.lock:
            for key, (started, _) in list(self.buckets.items()):
                if now - started >= 60:
                    del self.buckets[key]
            if address not in self.buckets and len(self.buckets) >= 2048:
                return False
            started, count = self.buckets.get(address, (now, 0))
            if count >= self.limit:
                return False
            self.buckets[address] = (started, count + 1)
            return True


def auth_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        # Never accept a token from query parameters, cookies or a client-supplied ownerId.
        header = request.headers.get("Authorization", "")
        parts = header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer" or len(parts[1]) > 8192:
            raise AuthError(401, "UNAUTHORIZED", "请先登录。")
        if not hasattr(g, "auth_user"):
            g.auth_user = client().userinfo(parts[1])
            g.access_token = parts[1]
            g.user_id = g.auth_user["sub"]
        return view(*args, **kwargs)

    return wrapped


def json_body(allowed: set[str]) -> dict:
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or set(data) - allowed:
        raise AuthError(400, "INVALID_REQUEST", "请求格式不正确。")
    return data


def text(data: dict, key: str, maximum: int, *, trim: bool = True) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise AuthError(400, "INVALID_REQUEST", "请填写完整的登录信息。")
    if trim:
        value = value.strip()
    if not value or len(value) > maximum:
        raise AuthError(400, "INVALID_REQUEST", "登录信息为空或超过长度限制。")
    return value


def token_response(result: dict):
    token = result.get("access_token")
    if (
        not isinstance(token, str)
        or not token
        or len(token) > 8192
        or any(c.isspace() for c in token)
    ):
        raise AuthError(502, "AUTH_BAD_RESPONSE", "认证服务未返回有效令牌。")
    # Explicit allowlist: never forward client_secret, password hashes or session_key.
    return {"code": 200, "access_token": token}


def public_user(info: dict):
    return {
        "id": info["sub"],
        "nickname": info.get("name") if isinstance(info.get("name"), str) else "草木观察者",
    }


def create_auth_blueprint() -> Blueprint:
    auth = Blueprint("auth", __name__, url_prefix="/api/auth")

    @auth.before_request
    def limit_attempts():
        if request.method == "POST":
            client().require_configured()
            limiter = current_app.extensions["auth_limiter"]
            if not limiter.allow(request.remote_addr or "unknown"):
                raise AuthError(429, "AUTH_RATE_LIMITED", "操作过于频繁，请稍后重试。")

    @auth.post("/login")
    def login():
        data = json_body({"username", "password"})
        result = client().login(
            text(data, "username", 128), text(data, "password", 256, trim=False)
        )
        return token_response(result)

    @auth.post("/register")
    def register():
        data = json_body({"username", "password", "nickname"})
        raw_nickname = data.get("nickname")
        if isinstance(raw_nickname, str) and not raw_nickname.strip():
            raw_nickname = None
        nickname = text(data, "nickname", 40) if raw_nickname is not None else None
        result = client().register(
            text(data, "username", 128),
            text(data, "password", 256, trim=False),
            nickname,
        )
        return token_response(result)

    @auth.post("/wechat")
    def wechat():
        data = json_body({"code"})
        return token_response(client().wechat_login(text(data, "code", 512)))

    @auth.get("/me")
    @auth_required
    def me():
        return {"code": 200, "user": public_user(g.auth_user)}

    @auth.post("/bind/wechat")
    @auth_required
    def bind_wechat():
        data = json_body({"code"})
        client().bind_wechat(g.access_token, text(data, "code", 512))
        return {"code": 200, "message": "微信身份绑定成功。"}

    return auth
