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
    r.take("ocr", "u")
    r.undo("ocr", "u")
    assert r.usage("ocr_day", "u")["used"] == 0
    assert r.usage("app_ocr_day", "app")["used"] == 0


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
