from utils import quota, security
from utils.security import limiter

BIG, SMALL = "openai/gpt-oss-120b", "openai/gpt-oss-20b"


def fresh(monkeypatch, per_user=2):
    shared = security.RateLimiter()
    monkeypatch.setattr(quota, "limiter", lambda: shared)
    security.register_model_limits({BIG: (per_user, 500), SMALL: (100, 1000), "qwen/qwen3.6-27b": (50, 500)})
    return shared


def test_using_up_one_model_locks_every_model_for_that_user(monkeypatch):
    fresh(monkeypatch)
    assert quota.reserve(BIG, "u")[0] and quota.reserve(BIG, "u")[0]
    ok, message = quota.reserve(SMALL, "u")
    assert not ok and "all models are paused" in message
    assert quota.reserve(SMALL, "someone_else")[0]  # other users unaffected


def test_provider_pause_blocks_only_that_model(monkeypatch):
    r = fresh(monkeypatch)
    r.block(f"model:{BIG}", 120)
    assert not quota.reserve(BIG, "u")[0]
    assert quota.reserve(SMALL, "u")[0]


def test_release_refunds_the_question(monkeypatch):
    r = fresh(monkeypatch)
    _, receipt = quota.reserve(SMALL, "u")
    quota.release(SMALL, "u", receipt)
    assert r.usage("question_day", "u")["used"] == 0
    assert r.usage(f"model_day:{SMALL}", "u")["used"] == 0


def test_best_model_prefers_most_capacity_and_skips_paused(monkeypatch):
    r = fresh(monkeypatch, per_user=10)
    assert quota.best_model("me") == BIG  # all fresh: ties go to the preferred default
    for _ in range(5):
        r.take(f"model:{BIG}", "others")  # other users spend some of 120B's app-wide budget
    assert quota.best_model("me") == SMALL  # 20B now has the most capacity left
    r.block(f"model:{SMALL}", 60)
    assert quota.best_model("me") != SMALL  # a paused model is never the default


def test_limiter_is_shared_by_default():
    assert limiter() is limiter()
