"""Server-only configuration. Authentication secrets never enter frontend builds."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from yizhu_server.recognition_solutions import DEFAULT_SOLUTION, get_solution

DEFAULT_CORS_ORIGINS = (
    "https://yizhuh5-h.942295.xyz",
    "https://yizhuh5.942295.xyz",
)


@dataclass(frozen=True)
class Settings:
    h5_dist: Path
    allowed_hosts: tuple[str, ...] = (
        "127.0.0.1",
        "localhost",
        "yizhu.942295.xyz",
        "yizhuh5.942295.xyz",
    )
    allowed_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    auth_issuer: str = "https://auth.942295.xyz"
    auth_client_id: str = ""
    auth_client_secret: str = field(default="", repr=False)
    auth_attempts_per_minute: int = 20
    data_dir: Path | None = None
    recognition_solution: str = DEFAULT_SOLUTION

    def __post_init__(self):
        get_solution(self.recognition_solution)

    @classmethod
    def from_env(cls) -> "Settings":
        repo = Path(__file__).resolve().parents[3]
        directory = Path(os.environ.get("YIZHU_H5_DIST", repo / "client/dist/h5"))
        origins = tuple(
            origin.strip()
            for origin in os.environ.get("CORS_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS)).split(",")
            if origin.strip()
        )
        if "*" in origins:
            raise ValueError("CORS_ORIGINS must list explicit trusted origins, not *")
        return cls(
            h5_dist=directory.expanduser().resolve(),
            allowed_origins=origins,
            auth_issuer=os.environ.get("ZDOG_AUTH_ISSUER", "https://auth.942295.xyz").rstrip("/"),
            auth_client_id=os.environ.get("ZDOG_AUTH_CLIENT_ID", "").strip(),
            auth_client_secret=os.environ.get("ZDOG_AUTH_CLIENT_SECRET", ""),
            data_dir=Path(os.environ.get("YIZHU_DATA_DIR", repo / "server/.data")).resolve(),
            recognition_solution=os.environ.get("YIZHU_RECOGNITION_SOLUTION", DEFAULT_SOLUTION).strip(),
        )
