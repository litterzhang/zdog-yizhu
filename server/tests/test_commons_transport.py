from datetime import UTC, datetime, timedelta

import pytest

from yizhu_server import commons_transport as transport

NOW = datetime(2026, 9, 25, 4, 30, tzinfo=UTC)


def test_retry_after_honors_seconds_and_http_dates():
    assert transport.retry_time("1800", NOW) == NOW + timedelta(seconds=1800)
    assert transport.retry_time("Fri, 25 Sep 2026 05:30:00 GMT", NOW) == NOW + timedelta(hours=1)
    for header in (None, "invalid", "10"):
        assert transport.retry_time(header, NOW) == NOW + timedelta(seconds=600)


def test_resume_requires_explicit_request_and_expired_cooldown():
    stopped = {
        "httpStatus": 429, "stoppedAt": NOW.isoformat(),
        "retryNotBefore": (NOW + timedelta(seconds=1200)).isoformat(),
    }
    with pytest.raises(ValueError, match="explicitly"):
        transport.require_resume(stopped, requested=False, current=NOW + timedelta(hours=1))
    with pytest.raises(ValueError, match="cooldown"):
        transport.require_resume(stopped, requested=True, current=NOW + timedelta(seconds=601))
    transport.require_resume(stopped, requested=True, current=NOW + timedelta(hours=1))
    with pytest.raises(ValueError, match="manual resolution"):
        transport.require_resume({**stopped, "httpStatus": 403}, requested=True, current=NOW)


def test_legacy_429_without_retry_header_gets_conservative_wait():
    stopped = {"httpStatus": 429, "stoppedAt": NOW.isoformat()}
    with pytest.raises(ValueError, match="cooldown"):
        transport.require_resume(stopped, requested=True, current=NOW + timedelta(seconds=599))
    transport.require_resume(stopped, requested=True, current=NOW + timedelta(seconds=601))


def test_connection_failure_needs_separate_explicit_opt_in_and_cooldown():
    stopped = {"errorType": "ConnectionError", "sourceRestriction": False,
               "stoppedAt": NOW.isoformat()}
    later = NOW + timedelta(hours=1)
    with pytest.raises(ValueError, match="manual resolution"):
        transport.require_resume(stopped, requested=True, current=later)
    with pytest.raises(ValueError, match="cooldown"):
        transport.require_resume(stopped, requested=True, current=NOW, allow_transport_retry=True)
    transport.require_resume(stopped, requested=True, current=later, allow_transport_retry=True)
    for changes in [
        {"httpStatus": 403}, {"sourceRestriction": True}, {"errorType": "ValueError"},
        {"sourceRestriction": None},
    ]:
        with pytest.raises(ValueError, match="manual resolution"):
            transport.require_resume({**stopped, **changes}, requested=True,
                                     current=later, allow_transport_retry=True)


class Response:
    def __init__(self, status=200, content=b'{"query": {}}', headers=None):
        self.status_code, self.content, self.headers = status, content, headers or {}
        self.is_redirect = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        assert self.status_code == 200

    def iter_content(self, size):
        yield self.content


def client(monkeypatch, response):
    instance = transport.CommonsClient()
    calls, sleeps = [], []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(instance.session, "get", get)
    monkeypatch.setattr(transport.time, "monotonic", lambda: 100.)
    monkeypatch.setattr(transport.time, "sleep", sleeps.append)
    return instance, calls, sleeps


def test_serial_identified_requests_include_maxlag_and_spacing(monkeypatch):
    instance, calls, sleeps = client(monkeypatch, Response())
    for _ in range(2):
        instance.get("https://commons.wikimedia.org/w/api.php", params={"action": "query"})
    assert calls[0][1]["params"]["maxlag"] == 5
    assert calls[0][1]["allow_redirects"] is False
    assert sleeps == [10.]
    assert instance.session.headers["User-Agent"] == transport.USER_AGENT


@pytest.mark.parametrize("status", [403, 429, 503])
def test_restrictions_stop_without_internal_retry(monkeypatch, status):
    instance, calls, _ = client(monkeypatch, Response(status, headers={"Retry-After": "1800"}))
    with pytest.raises(transport.SourceRestriction) as error:
        instance.get("https://commons.wikimedia.org/w/api.php")
    assert error.value.status == status
    assert error.value.retry_after == "1800"
    assert bool(error.value.not_before) == (status != 403)
    assert len(calls) == 1


def test_maxlag_json_and_response_size_fail_closed(monkeypatch):
    instance, _, _ = client(monkeypatch, Response(content=b'{"error":{"code":"maxlag"}}'))
    with pytest.raises(transport.SourceRestriction):
        instance.get("https://commons.wikimedia.org/w/api.php")
    instance, _, _ = client(monkeypatch, Response(content=b"123456"))
    with pytest.raises(ValueError, match="size limit"):
        instance.get("https://upload.wikimedia.org/wikipedia/commons/a.jpg", limit=2)


def test_unapproved_hosts_never_contacted(monkeypatch):
    instance, calls, _ = client(monkeypatch, Response())
    with pytest.raises(ValueError, match="official HTTPS"):
        instance.get("https://localhost/internal")
    assert calls == []
