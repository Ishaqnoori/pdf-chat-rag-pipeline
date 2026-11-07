import time

from utils import monitoring, session, store

USER = {"uid": "u1", "email": "a@b.com", "name": "Ishaq"}


def fake_store(monkeypatch, active=True):
    monkeypatch.setattr(session, "_cached_secret", None)
    monkeypatch.setattr(session, "_browser", lambda: "same-browser")
    monkeypatch.setattr(store, "enabled", lambda: True)
    monkeypatch.setattr(store, "session_secret", lambda: "s" * 48)
    monkeypatch.setattr(store, "create_session", lambda *a: None)
    monkeypatch.setattr(store, "session_active", lambda uid, sid: active)


def test_valid_token_round_trip(monkeypatch):
    fake_store(monkeypatch)
    token, sid = session.issue(USER)
    data = session.verify(token)
    assert data["uid"] == "u1" and data["sid"] == sid


def test_tampered_token_is_rejected(monkeypatch):
    fake_store(monkeypatch)
    token, _ = session.issue(USER)
    payload, sig = token.split(".")
    forged = session._b64(session._unb64(payload).replace(b'"u1"', b'"admin"')) + "." + sig
    assert session.verify(forged) is None


def test_revoked_session_is_rejected(monkeypatch):
    fake_store(monkeypatch, active=False)  # e.g. the user logged out
    token, _ = session.issue(USER)
    assert session.verify(token) is None


def test_expired_token_is_rejected(monkeypatch):
    fake_store(monkeypatch)
    real_time = time.time
    token, _ = session.issue(USER)
    monkeypatch.setattr(session.time, "time", lambda: real_time() + session.LIFETIME + 10)
    assert session.verify(token) is None


def test_cookie_copied_to_another_browser_is_rejected(monkeypatch):
    fake_store(monkeypatch)
    monkeypatch.setattr(session, "_browser", lambda: "chrome-on-my-laptop")
    token, _ = session.issue(USER)
    monkeypatch.setattr(session, "_browser", lambda: "attackers-browser")
    assert session.verify(token) is None


def test_signing_everyone_out_invalidates_old_tokens(monkeypatch):
    fake_store(monkeypatch)
    token, _ = session.issue(USER)
    monkeypatch.setattr(store, "rotate_session_secret", lambda: True)
    monkeypatch.setattr(store, "session_secret", lambda: "n" * 48)  # the new secret
    assert session.sign_everyone_out()
    assert session.verify(token) is None


def test_sentry_events_are_scrubbed():
    event = {"request": {"cookies": "x"}, "user": {"ip_address": "1.2.3.4"},
             "exception": {"values": [{"stacktrace": {"frames": [{"vars": {"question": "secret"}}]}}]}}
    clean = monitoring._scrub(event, None)
    assert "request" not in clean and "user" not in clean
    assert "vars" not in clean["exception"]["values"][0]["stacktrace"]["frames"][0]
