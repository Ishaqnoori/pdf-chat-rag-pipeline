import time

from utils import auth

GOOD = dict(name="Ishaq", email="a@b.com", password="password1", confirm="password1")


def test_sign_up_requires_agreeing_to_the_terms():
    ok, message = auth.sign_up(**GOOD, agreed=False)
    assert not ok and "Terms of Service" in message


def test_honeypot_blocks_bots(monkeypatch):
    called = []
    monkeypatch.setattr(auth.fb, "sign_up", lambda *a: called.append(a))
    ok, _ = auth.sign_up(**GOOD, agreed=True, honeypot="http://spam.example")
    assert not ok and not called  # refused before contacting Firebase


def test_instant_submissions_are_refused(monkeypatch):
    called = []
    monkeypatch.setattr(auth.fb, "sign_up", lambda *a: called.append(a))
    ok, _ = auth.sign_up(**GOOD, agreed=True, form_shown_at=time.time())
    assert not ok and not called
