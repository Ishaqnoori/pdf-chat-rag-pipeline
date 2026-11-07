import time

from utils import security
from utils.security import RateLimiter, clean_text, register_model_limits, sanitize_markdown


def test_take_refuses_when_full_and_does_not_count_refusals():
    r = RateLimiter()
    limit = security.LIMITS["upload"][0]
    for _ in range(limit):
        assert r.take("upload", "alice")[0]
    ok, message = r.take("upload", "alice")
    assert not ok and "limit" in message.lower()
    assert r.usage("upload", "alice")["used"] == limit
    assert r.take("upload", "bob")[0]  # other users are unaffected


def test_undo_refunds_every_bucket():
    r = RateLimiter()
    _, receipt = r.take("ocr", "u")
    r.undo("ocr", "u", receipt)
    assert r.usage("ocr_day", "u")["used"] == 0
    assert r.usage("app_ocr_day", "app")["used"] == 0


def test_refund_removes_exactly_its_own_event():
    r = RateLimiter()
    _, first = r.take("ocr", "u")
    time.sleep(0.01)
    _, second = r.take("ocr", "u")  # another request recorded after the first
    r.undo("ocr", "u", first)       # the first one fails and is refunded
    assert list(r._hits["ocr_day:u"]) == [second]


def test_model_limits_are_per_model():
    register_model_limits({"m1": (1, 100), "m2": (5, 100)})
    r = RateLimiter()
    assert r.take("model:m1", "u", label="M1")[0]
    ok, message = r.take("model:m1", "u", label="M1")
    assert not ok and message.startswith("M1")
    assert r.take("model:m2", "u")[0]


def test_block_pauses_for_everyone():
    r = RateLimiter()
    r.block("model:m1", 30)
    assert 0 < r.blocked_for("model:m1") <= 30
    assert r.blocked_for("model:m2") == 0


def test_sanitize_strips_html_images_and_disguised_links():
    out = sanitize_markdown('Hi ![x](http://evil/p?d=1) <script>x</script> [click](https://evil.com) **ok**')
    assert "<script>" not in out and "![" not in out
    assert "click (https://evil.com)" in out and "**ok**" in out


def test_clean_text_removes_control_chars_and_trims():
    assert clean_text("  a\x07b\x00  ", 10) == "ab"
    assert len(clean_text("x" * 50, 10)) == 10


def test_counters_survive_a_restart():
    r = RateLimiter()
    for _ in range(3):
        r.take("question", "alice")
    r.block("model:x", 300)
    restored = RateLimiter()
    restored.load(r.export())
    assert restored.usage("question_day", "alice")["used"] == 3
    assert restored.blocked_for("model:x") > 290


def test_dangerous_link_schemes_become_plain_text():
    for attack in ("[click](javascript:alert(1))", "[x](JaVaScRiPt:alert`1`)", "[p](data:text/html;base64,PHM+)",
                   "<javascript:alert(1)>", "<img src=x onerror=alert(1)>"):
        out = sanitize_markdown(attack)
        assert "javascript" not in out.lower() and "data:" not in out and "<" not in out, attack


def test_web_links_show_their_address():
    assert sanitize_markdown('[docs](https://x.com/a_(b) "t")') == "docs (https://x.com/a_(b))"
