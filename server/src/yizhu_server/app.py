"""Flask API with zdog-auth; Nginx serves the production H5 build directly."""

import re
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file
from werkzeug.exceptions import HTTPException

from yizhu_server.auth import AttemptLimiter, create_auth_blueprint
from yizhu_server.authclient import AuthClient, AuthError
from yizhu_server.settings import Settings

PUBLIC_EXTENSIONS = {
    ".html",
    ".js",
    ".css",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".svg",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".json",
}
HASHED_ASSET = re.compile(r"\.[a-f0-9]{8,}\.(?:js|css|woff2?)$")
CORS_METHODS = {"GET", "HEAD", "POST", "OPTIONS"}
CORS_HEADERS = {"authorization", "content-type", "x-photo-id"}


def create_app(
    settings: Settings | None = None, *, auth_client: AuthClient | None = None,
    recognition_runtime=None,
) -> Flask:
    settings = settings or Settings.from_env()
    root = settings.h5_dist.resolve()
    app = Flask(__name__, static_folder=None)
    app.config.update(
        DEBUG=False,
        TRUSTED_HOSTS=list(settings.allowed_hosts),
        MAX_CONTENT_LENGTH=16 * 1024,
    )
    app.json.ensure_ascii = False
    app.extensions["zdog_auth"] = auth_client or AuthClient(settings)
    app.extensions["auth_limiter"] = AttemptLimiter(settings.auth_attempts_per_minute)
    if settings.data_dir is not None or recognition_runtime is not None:
        from yizhu_server.journal_api import RecognitionRuntime, create_journal_blueprint
        from yizhu_server.journal_store import JournalStore

        if settings.data_dir and settings.data_dir.is_relative_to(root):
            raise ValueError("Private data directory must not be inside the public H5 root")
        runtime = recognition_runtime or RecognitionRuntime(
            JournalStore(settings.data_dir), solution_id=settings.recognition_solution,
        )
        app.extensions["recognition_runtime"] = runtime
        app.register_blueprint(create_journal_blueprint(runtime))

    @app.before_request
    def cors():
        if not request.path.startswith("/api/"):
            return None
        if request.path == "/api/photos" and request.method == "POST":
            request.max_content_length = 1024 * 1024
            request.max_form_parts = 4
        origin = request.headers.get("Origin")
        if request.method == "OPTIONS":
            wanted_headers = {
                value.strip().lower()
                for value in request.headers.get("Access-Control-Request-Headers", "").split(",")
                if value.strip()
            }
            if (
                origin not in settings.allowed_origins
                or request.headers.get("Access-Control-Request-Method", "GET") not in CORS_METHODS
                or not wanted_headers.issubset(CORS_HEADERS)
            ):
                return {"code": 403, "message": "不允许此跨域请求。"}, 403
            return "", 204
        if (
            request.method not in {"GET", "HEAD"}
            and origin
            and origin not in settings.allowed_origins
        ):
            return {"code": 403, "message": "不允许此跨域请求。"}, 403
        return None

    @app.after_request
    def response_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
            origin = request.headers.get("Origin")
            if origin in settings.allowed_origins:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, POST, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, X-Photo-Id"
                response.vary.add("Origin")
        return response

    @app.errorhandler(AuthError)
    def auth_error(error):
        response = jsonify(code=error.status, error=error.code, message=error.message)
        response.status_code = error.status
        if error.status == 429:
            response.headers["Retry-After"] = "60"
        return response

    @app.errorhandler(HTTPException)
    def http_error(error):
        messages = {
            400: "请求格式不正确。",
            404: "接口或文件不存在。",
            405: "不支持此请求方式。",
            413: "请求内容过大。",
        }
        return {"code": error.code, "message": messages.get(error.code, "请求未完成。")}, error.code

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "service": "yizhu-server",
            "framework": "flask",
            "mode": "live" if "recognition_runtime" in app.extensions else "mock",
            "recognition": app.extensions["recognition_runtime"].state if "recognition_runtime" in app.extensions else "disabled",
            "recognition_solution": app.extensions["recognition_runtime"].solution_info() if "recognition_runtime" in app.extensions else None,
            "h5_ready": (root / "index.html").is_file(),
            "auth": {"provider": "zdog-auth", "configured": app.extensions["zdog_auth"].configured},
        }

    app.register_blueprint(create_auth_blueprint())

    @app.get("/", defaults={"path": ""})
    @app.get("/<path:path>")
    def frontend(path: str):
        if (
            path == "api"
            or path.startswith("api/")
            or "\\" in path
            or any(segment.startswith(".") for segment in path.split("/"))
        ):
            abort(404)
        candidate = (root / path).resolve()
        if not candidate.is_relative_to(root):
            abort(404)
        if candidate.is_file():
            if candidate.suffix.lower() not in PUBLIC_EXTENSIONS:
                abort(404)
            cache = "no-cache"
            if HASHED_ASSET.search(candidate.name):
                cache = "public, max-age=31536000, immutable"
            elif candidate.suffix.lower() != ".html":
                cache = "public, max-age=3600"
            response = send_file(candidate)
            response.headers["Cache-Control"] = cache
            return response
        wants_document = not path or (
            path.startswith("pages/")
            and not Path(path).suffix
            and "text/html" in request.headers.get("Accept", "")
        )
        if not wants_document:
            abort(404)
        index = (root / "index.html").resolve()
        if not index.is_relative_to(root):
            abort(404)
        if not index.is_file():
            return {
                "code": 503,
                "message": "H5 build is missing. Run npm --prefix client run build:h5.",
            }, 503
        response = send_file(index)
        response.headers["Cache-Control"] = "no-cache"
        return response

    return app
