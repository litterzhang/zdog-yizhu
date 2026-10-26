"""Serial, identified Commons requests with persistent server-directed backoff."""

import json
import time
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import requests

USER_AGENT = "YizhuPlantResearch/0.1 (+https://yizhuh5-h.942295.xyz; offline dataset collection)"
MIN_INTERVAL = 10.
DEFAULT_COOLDOWN = 600
ALLOWED_HOSTS = {"commons.wikimedia.org", "upload.wikimedia.org", "thumb.wikimedia.org"}


def retry_time(header, current):
    """Use the longer of the server wait and conservative local cooldown."""
    fallback = current + timedelta(seconds=DEFAULT_COOLDOWN)
    if not header:
        return fallback
    try:
        if header.strip().isdigit():
            requested = current + timedelta(seconds=int(header.strip()))
        else:
            requested = parsedate_to_datetime(header)
            if requested.tzinfo is None:
                requested = requested.replace(tzinfo=UTC)
        return max(fallback, requested)
    except (ValueError, TypeError, OverflowError):
        return fallback


class SourceRestriction(PermissionError):
    def __init__(self, status, header=None):
        super().__init__(f"Commons request stopped: HTTP {status}")
        self.status = status
        self.not_before = retry_time(header, datetime.now(UTC)).isoformat() if status != 403 else None
        self.retry_after = header


def require_resume(stopped, *, requested, current=None, allow_transport_retry=False):
    """A stopped run never resumes merely because somebody ran the command again."""
    if not stopped:
        return
    if not requested:
        raise ValueError("Prior run stopped; inspect it and explicitly request resume.")
    transport_failure = (
        allow_transport_retry
        and stopped.get("sourceRestriction") is False
        and stopped.get("httpStatus") is None
        and stopped.get("errorType") in {"ConnectionError", "ConnectTimeout", "ReadTimeout"}
    )
    if stopped.get("httpStatus") not in {429, 503} and not transport_failure:
        raise ValueError("Non-rate restrictions need manual resolution; do not bypass.")
    current = current or datetime.now(UTC)
    not_before = stopped.get("retryNotBefore")
    if not_before:
        deadline = datetime.fromisoformat(not_before)
    else:
        deadline = datetime.fromisoformat(stopped["stoppedAt"]) + timedelta(seconds=DEFAULT_COOLDOWN)
    if deadline.tzinfo is None:
        raise ValueError("Cooldown time must include timezone.")
    if current < deadline:
        raise ValueError(f"Respect source cooldown until {deadline.isoformat()}.")


class CommonsClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.last_finished = None

    def get(self, url, *, params=None, limit=12 * 1024 * 1024):
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS
            or parsed.username or parsed.password or parsed.port not in {None, 443}
        ):
            raise ValueError("Commons client accepts only official HTTPS hosts.")
        api = parsed.hostname == "commons.wikimedia.org"
        if self.last_finished is not None:
            time.sleep(max(0., MIN_INTERVAL - (time.monotonic() - self.last_finished)))
        try:
            with self.session.get(
                url, params={**(params or {}), "maxlag": 5} if api else params,
                timeout=(15, 90), stream=True, allow_redirects=False,
            ) as response:
                if response.status_code in {403, 429, 503}:
                    raise SourceRestriction(response.status_code, response.headers.get("Retry-After"))
                if response.is_redirect:
                    raise ValueError("Unexpected Commons redirect; do not follow to an unknown host.")
                response.raise_for_status()
                data = bytearray()
                for chunk in response.iter_content(65536):
                    data.extend(chunk)
                    if len(data) > limit:
                        raise ValueError("Commons response exceeds size limit.")
                if api:
                    payload = json.loads(data)
                    if payload.get("error", {}).get("code") == "maxlag":
                        raise SourceRestriction(503, response.headers.get("Retry-After"))
                return bytes(data)
        finally:
            self.last_finished = time.monotonic()
