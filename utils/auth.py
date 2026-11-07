"""
Login, sign-up and session handling on top of Firebase Authentication.
All checks run on the server (item 6); the browser only ever sees the result.
"""
import time

import streamlit as st

from utils import firebase_auth as fb
from utils import session, store
from utils.security import (
    MAX_NAME, MAX_PASSWORD, MIN_PASSWORD, clean_text, client_ip, limiter, log, valid_email, wait_text,
)

IDLE_TIMEOUT = 30 * 60  # seconds of inactivity before automatic logout (item 9)


def google_configured() -> bool:
    """True when .streamlit/secrets.toml has a real [auth.google] section for st.login."""
    try:
        cid = str(st.secrets["auth"]["google"]["client_id"])
        secret = str(st.secrets["auth"]["cookie_secret"])
        return cid.endswith(".apps.googleusercontent.com") and len(secret) >= 32
    except Exception:
        return False


def _google_user():
    """Reads the signed-in Google identity from Streamlit's secure cookie, if any."""
    if not google_configured():
        return None
    try:
        if not st.user.is_logged_in or not st.user.get("email_verified", False):
            return None
        email = clean_text(st.user.get("email"), 254).lower()
        name = clean_text(st.user.get("name") or email.split("@")[0], MAX_NAME)
        return {"name": name, "email": email, "uid": "google:" + str(st.user.get("sub", email)), "method": "google"}
    except Exception:
        return None


# ---------------------------------------------------------------- session
def current_user():
    """Returns {"name", "email", "uid"} or None. Also enforces the idle timeout."""
    user = st.session_state.get("auth_user")
    if not user:
        user = _google_user() or _remembered_user()
        if not user:
            return None
        st.session_state.auth_user = user
        st.session_state.auth_last_seen = time.time()
    now = time.time()
    # "Keep me signed in" turns off the idle timeout; otherwise 30 idle minutes log you out
    if not user.get("remember") and now - st.session_state.get("auth_last_seen", now) > IDLE_TIMEOUT:
        logout()
        st.session_state.auth_notice = "You were logged out after 30 minutes of inactivity."
        return None
    st.session_state.auth_last_seen = now
    return user


def _remembered_user():
    """Signs a returning email user back in from their "keep me signed in" cookie."""
    data = session.from_browser()
    if not data:
        return None
    return {"name": clean_text(data["name"], MAX_NAME), "email": data["email"], "uid": data["uid"],
            "method": "firebase", "sid": data["sid"], "remember": True}


def logout():
    user = st.session_state.get("auth_user") or {}
    notice = st.session_state.get("auth_notice")
    if user.get("sid"):
        store.end_session(user["uid"], user["sid"])  # revoked on the server, even if the cookie lingers
    for k in list(st.session_state.keys()):
        del st.session_state[k]  # clears the chat and the document too
    if notice:
        st.session_state.auth_notice = notice
    if user.get("sid"):
        st.session_state.clear_cookie = True  # the login page removes it from the browser
    if user.get("method") == "google":
        st.logout()  # also clears Streamlit's Google login cookie


# ---------------------------------------------------------------- actions (each returns (ok, message))
def login(email: str, password: str, remember: bool = False):
    email = clean_text(email, 254).lower()
    password = str(password or "")[:MAX_PASSWORD]
    if not valid_email(email) or not password:
        return False, "Enter your email and password."

    rl, ip = limiter(), client_ip()
    for kind, key in (("login_email", email), ("login_ip", ip)):
        ok, wait = rl.check(kind, key, record=False)
        if not ok:
            return False, f"Too many attempts. Try again in {wait_text(wait)}."

    try:
        u = fb.sign_in(email, password)
    except fb.FirebaseError as e:
        if str(e) == fb.BAD_CREDENTIALS:
            rl.record("login_email", email)
            rl.record("login_ip", ip)
            log.info("Failed login for a user")  # never log the email or password
        return False, str(e)

    if not u["verified"]:
        ok, _ = rl.check("email_send", email)
        if ok:
            try:
                fb.send_verification(u["token"])
            except fb.FirebaseError:
                pass
        return False, "Please verify your email first. We've sent you a new verification link."

    rl.reset("login_email", email)
    user = {"name": clean_text(u["name"], MAX_NAME), "email": u["email"], "uid": u["uid"], "method": "firebase"}
    if remember and session.available():
        issued = session.issue(user)
        if issued:
            st.session_state.pending_cookie, user["sid"] = issued  # written to the browser on the next page
            user["remember"] = True
    st.session_state.auth_user = user
    st.session_state.auth_last_seen = time.time()
    return True, None


MIN_FORM_SECONDS = 3  # people take longer than this to fill in the sign-up form; scripts don't


def sign_up(name: str, email: str, password: str, confirm: str, agreed: bool = False,
            honeypot: str = "", form_shown_at: float | None = None):
    """
    Creates an account. Bot checks: a hidden field people never see (honeypot) must stay empty,
    and the form can't be submitted faster than a person could fill it in.
    """
    if honeypot or (form_shown_at and time.time() - form_shown_at < MIN_FORM_SECONDS):
        log.info("Sign-up refused by bot check")
        return False, "Something went wrong. Please try again in a moment."  # don't explain to bots
    if not agreed:
        return False, "Please agree to the Terms of Service and Privacy Policy to create an account."
    name = clean_text(name, MAX_NAME + 1)
    email = clean_text(email, 255).lower()
    password, confirm = str(password or ""), str(confirm or "")

    if not name:
        return False, "Enter your name."
    if len(name) > MAX_NAME:
        return False, f"Name must be {MAX_NAME} characters or fewer."
    if not valid_email(email):
        return False, "Enter a valid email address."
    if not (MIN_PASSWORD <= len(password) <= MAX_PASSWORD):
        return False, f"Password must be {MIN_PASSWORD}–{MAX_PASSWORD} characters."
    if password != confirm:
        return False, "Passwords don't match."

    ok, wait = limiter().check("signup_ip", client_ip())
    if not ok:
        return False, f"Too many new accounts from this network. Try again in {wait_text(wait)}."

    try:
        fb.sign_up(name, email, password)
    except fb.FirebaseError as e:
        return False, str(e)
    return True, f"Account created. We've sent a verification link to {email}. Open it, then log in."


def send_reset(email: str):
    email = clean_text(email, 254).lower()
    if not valid_email(email):
        return False, "Enter a valid email address."
    generic = f"If an account exists for {email}, a reset link is on its way."
    ok, _ = limiter().check("email_send", email)
    if not ok:
        return True, generic  # same answer either way: no hints about which emails exist
    try:
        fb.send_password_reset(email)
    except fb.FirebaseError as e:
        if str(e) != fb.BAD_CREDENTIALS:
            return False, str(e)
    return True, generic


def delete_account(password: str | None = None):
    """
    Permanently deletes the signed-in user's account and all their saved data.
    Email accounts must confirm with their password. Returns (ok, message).
    """
    user = st.session_state.get("auth_user") or {}
    if not user:
        return False, "You're not signed in."
    if user.get("method") == "firebase":
        if not password:
            return False, "Enter your password to confirm."
        try:
            u = fb.sign_in(user["email"], password)  # fresh sign-in proves it's really you
            fb.delete_account(u["token"])
        except fb.FirebaseError as e:
            return False, str(e)
    store.delete_user_data(str(user["uid"]).replace("/", "_"))
    log.info("Account deleted")
    st.session_state.auth_notice = "Your account and saved data have been deleted."
    logout()
    return True, None


def initials(name: str) -> str:
    parts = [p for p in (name or "").replace(".", " ").split() if p]
    if not parts:
        return "?"
    return (parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper()
