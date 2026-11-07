"""
Firebase Authentication (email + password) via the Identity Toolkit REST API.
Needs FIREBASE_API_KEY (Firebase console > Project settings > Your apps > Config > apiKey).
Firebase stores and hashes the passwords (item 10); we never keep them.
"""
import os

import requests

from utils.security import log

BASE = "https://identitytoolkit.googleapis.com/v1/accounts"
TIMEOUT = 10

FRIENDLY_ERRORS = {
    "EMAIL_EXISTS": "An account with this email already exists. Log in instead.",
    "INVALID_EMAIL": "Enter a valid email address.",
    "WEAK_PASSWORD": "Password is too weak. Use at least 8 characters.",
    "PASSWORD_DOES_NOT_MEET_REQUIREMENTS":
        "Password doesn't meet the requirements. Try a longer one with mixed characters.",
    "INVALID_LOGIN_CREDENTIALS": "Incorrect email or password.",
    "EMAIL_NOT_FOUND": "Incorrect email or password.",
    "INVALID_PASSWORD": "Incorrect email or password.",
    "USER_DISABLED": "This account has been disabled.",
    "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many attempts. Try again in a few minutes.",
}
BAD_CREDENTIALS = FRIENDLY_ERRORS["INVALID_LOGIN_CREDENTIALS"]


class FirebaseError(Exception):
    pass


def api_key():
    key = (os.getenv("FIREBASE_API_KEY") or "").strip()
    return key if key and not key.lower().startswith("your") else None


def is_configured() -> bool:
    return api_key() is not None


def _post(endpoint: str, payload: dict) -> dict:
    try:
        r = requests.post(f"{BASE}:{endpoint}", params={"key": api_key()}, json=payload, timeout=TIMEOUT)
        data = r.json() if r.content else {}
    except (requests.RequestException, ValueError):
        log.warning("Firebase %s: network error", endpoint)
        raise FirebaseError("Couldn't reach the login service. Check your connection and try again.")
    if r.status_code != 200:
        code = str(data.get("error", {}).get("message", "UNKNOWN"))
        key = code.split(" ")[0].split(":")[0].strip()
        log.info("Firebase %s failed: %s", endpoint, key)  # code only, never the payload
        # Unknown codes (bad config etc.) get a generic message (item 17)
        raise FirebaseError(FRIENDLY_ERRORS.get(key, "Login is temporarily unavailable. Try again later."))
    return data


def sign_up(name: str, email: str, password: str) -> None:
    """Creates the account, sets the display name and sends a verification email."""
    data = _post("signUp", {"email": email, "password": password, "returnSecureToken": True})
    token = data["idToken"]
    _post("update", {"idToken": token, "displayName": name, "returnSecureToken": False})
    _post("sendOobCode", {"requestType": "VERIFY_EMAIL", "idToken": token})


def sign_in(email: str, password: str) -> dict:
    """Returns {"name", "email", "uid", "verified", "token"}."""
    data = _post("signInWithPassword", {"email": email, "password": password, "returnSecureToken": True})
    info = _post("lookup", {"idToken": data["idToken"]}).get("users", [{}])[0]
    email = data.get("email", email)
    return {
        "name": data.get("displayName") or info.get("displayName") or email.split("@")[0],
        "email": email,
        "uid": data.get("localId"),
        "verified": bool(info.get("emailVerified")),
        "token": data["idToken"],
    }


def send_verification(id_token: str) -> None:
    _post("sendOobCode", {"requestType": "VERIFY_EMAIL", "idToken": id_token})


def delete_account(id_token: str) -> None:
    """Permanently deletes the signed-in Firebase account."""
    _post("delete", {"idToken": id_token})


def send_password_reset(email: str) -> None:
    _post("sendOobCode", {"requestType": "PASSWORD_RESET", "email": email})
