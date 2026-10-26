"""zdog-auth machine API adapter, based on zdog-mp's current contract.

No local password database, self-signed JWT, startup provisioning or mock fallback.
"""

from urllib.parse import urlsplit

import requests

from yizhu_server.settings import Settings


class AuthError(RuntimeError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class AuthClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        issuer = urlsplit(settings.auth_issuer)
        secure = issuer.scheme == "https" or (
            issuer.scheme == "http" and issuer.hostname in {"localhost", "127.0.0.1", "::1"}
        )
        self.configured = bool(
            settings.auth_client_id
            and settings.auth_client_secret
            and secure
            and issuer.hostname
            and not (issuer.username or issuer.password or issuer.query or issuer.fragment)
        )

    def require_configured(self):
        if not self.configured:
            raise AuthError(503, "AUTH_NOT_CONFIGURED", "统一登录尚未配置，请联系管理员。")

    def _post(self, path: str, payload: dict) -> dict:
        self.require_configured()
        body = {
            **payload,
            "client_id": self.settings.auth_client_id,
            "client_secret": self.settings.auth_client_secret,
        }
        try:
            response = requests.post(
                f"{self.settings.auth_issuer.rstrip('/')}{path}",
                json=body,
                timeout=(3, 8),
                allow_redirects=False,
            )
        except requests.RequestException:
            # Never echo exceptions: they may contain URLs or request metadata.
            raise AuthError(503, "AUTH_UNAVAILABLE", "认证服务暂时不可用，请稍后重试。") from None

        if response.status_code >= 500:
            raise AuthError(503, "AUTH_UNAVAILABLE", "认证服务暂时不可用，请稍后重试。")
        if response.status_code == 429:
            raise AuthError(429, "AUTH_RATE_LIMITED", "操作过于频繁，请稍后重试。")
        if response.status_code >= 400:
            messages = {
                400: "登录信息无效或已过期，请检查后重试。",
                401: "登录凭据无效，请重新登录或检查用户名和密码。",
                403: "当前账号无法执行此操作。",
                409: "账号或微信绑定已存在，请登录已有账号。",
            }
            status = response.status_code if response.status_code in messages else 400
            raise AuthError(status, "AUTH_REJECTED", messages[status])
        if not 200 <= response.status_code < 300:
            raise AuthError(502, "AUTH_BAD_RESPONSE", "认证服务返回异常，请稍后重试。")
        try:
            data = response.json()
        except ValueError:
            raise AuthError(502, "AUTH_BAD_RESPONSE", "认证服务返回异常，请稍后重试。") from None
        if not isinstance(data, dict) or data.get("error"):
            raise AuthError(502, "AUTH_BAD_RESPONSE", "认证服务返回异常，请稍后重试。")
        return data

    def login(self, username: str, password: str) -> dict:
        return self._post("/machine/password/login", {"username": username, "password": password})

    def register(self, username: str, password: str, nickname: str | None = None) -> dict:
        return self._post(
            "/machine/register",
            {
                "username": username,
                "password": password,
                "nickname": nickname,
            },
        )

    def wechat_login(self, code: str) -> dict:
        return self._post("/machine/wechat/miniprogram", {"code": code})

    def userinfo(self, token: str) -> dict:
        info = self._post("/machine/userinfo", {"access_token": token})
        if not isinstance(info.get("sub"), str) or not info["sub"].strip():
            raise AuthError(401, "UNAUTHORIZED", "登录已失效，请重新登录。")
        return info

    def bind_wechat(self, token: str, code: str) -> dict:
        return self._post("/machine/wechat/bind", {"access_token": token, "code": code})
