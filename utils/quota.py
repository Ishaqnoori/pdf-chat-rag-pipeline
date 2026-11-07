"""
Model quotas for one user: which models they can use right now, and reserving/refunding a question.

Rules
  - Each model has its own daily allowance per user and for the whole app (utils/llm.py).
  - If a user uses up their allowance on any one model, every model is paused for them until it
    resets, so switching models can't get around the limits.
  - A model the provider rate-limited is paused for everyone until the provider's reset time.
  - The app never switches model behind the user's back.
"""
from utils.llm import AVAILABLE_MODELS, MODEL_DAILY_LIMITS, default_model
from utils.security import duration_text, limiter, register_model_limits, wait_text

register_model_limits({m: MODEL_DAILY_LIMITS.get(m, (50, 500)) for m in AVAILABLE_MODELS})


def model_name(model: str) -> str:
    return AVAILABLE_MODELS.get(model, (model,))[0]


def paused_for(model: str) -> int:
    """Seconds until a model the provider rate-limited can be used again (0 = available)."""
    return limiter().blocked_for(f"model:{model}")


def headroom(model: str, user_key: str) -> float:
    """Share of capacity left (0..1): the tighter of the app-wide budget and the user's allowance; -1 if paused."""
    if paused_for(model):
        return -1.0
    shares = []
    for kind, key in ((f"app_model_day:{model}", "app"), (f"model_day:{model}", user_key)):
        u = limiter().usage(kind, key)
        shares.append((u["limit"] - u["used"]) / u["limit"])
    return min(shares)


def best_model(user_key: str) -> str:
    """The model with the most capacity left; ties go to the preferred default (GROQ_MODEL)."""
    preferred = default_model()
    order = sorted(AVAILABLE_MODELS, key=lambda m: m != preferred)  # preferred first, then menu order
    return max(order, key=lambda m: headroom(m, user_key))  # max() keeps the first of equal scores


def user_lock(user_key: str):
    """(model, seconds) if the user has used up their own allowance on any model; else None."""
    for model in AVAILABLE_MODELS:
        u = limiter().usage(f"model_day:{model}", user_key)
        if u["used"] >= u["limit"]:
            return model, u["next_free_in"]
    return None


def lock_message(lock) -> str:
    model, wait = lock
    return (f"You've reached your daily limit on {model_name(model)}, so all models are paused for you "
            f"until it resets in {wait_text(wait)}.")


def reserve(model: str, user_key: str):
    """
    Takes one question from the user's overall and per-model allowance.
    Returns (False, message) when refused, or (True, receipt) to pass to release() if no answer comes.
    """
    lock = user_lock(user_key)
    if lock:
        return False, lock_message(lock)
    paused = paused_for(model)
    if paused:
        return False, f"{model_name(model)} has reached its usage limit. It resets in {wait_text(paused)}."
    ok, model_receipt = limiter().take(f"model:{model}", user_key, label=model_name(model))
    if not ok:
        return False, model_receipt
    ok, question_receipt = limiter().take("question", user_key)
    if not ok:
        limiter().undo(f"model:{model}", user_key, model_receipt)  # nothing was asked: give it back
        return False, question_receipt
    return True, (model_receipt, question_receipt)


def release(model: str, user_key: str, receipt):
    """Refunds exactly the question reserve() took, when no answer could be produced."""
    if receipt:
        model_receipt, question_receipt = receipt
        limiter().undo(f"model:{model}", user_key, model_receipt)
        limiter().undo("question", user_key, question_receipt)


def caption(model: str, user_key: str) -> str:
    """Model description for the menu, or why it can't be used right now."""
    paused = paused_for(model)
    if paused:
        return f"Limit reached, resets in {duration_text(paused)}"
    u = limiter().usage(f"model_day:{model}", user_key)
    if u["used"] >= u["limit"]:
        return f"Daily limit reached, resets in {duration_text(u['full_reset_in'])}"
    return AVAILABLE_MODELS[model][1]
