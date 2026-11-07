"""
Writes the "Continue with Google" settings into .streamlit/secrets.toml.
Keeps any other settings already in the file and generates the cookie secret for you.
Run with:  setup_google_login.bat
"""
import getpass
import json
import os
import secrets
import tomllib

PATH = os.path.join(".streamlit", "secrets.toml")
REDIRECT = "http://localhost:8501/oauth2callback"
METADATA = "https://accounts.google.com/.well-known/openid-configuration"


def ask(prompt, check, hidden=False):
    while True:
        value = (getpass.getpass(prompt) if hidden else input(prompt)).strip().strip('"').strip("'")
        if check(value):
            return value
        print("  That doesn't look right. Copy it again with the copy button and paste.\n")


print("Paste the values from Google Auth Platform > Clients > your client.\n")
client_id = ask("Client ID: ", lambda v: v.endswith(".apps.googleusercontent.com") and " " not in v)
print("Client secret: when you paste, nothing will appear on screen. That's normal; just press Enter.")
client_secret = ask("Client secret: ", lambda v: v.startswith("GOCSPX-") and " " not in v, hidden=True)

# Keep existing top-level settings (API keys etc.), replace any old [auth] section
existing = {}
if os.path.exists(PATH):
    try:
        with open(PATH, "rb") as f:
            existing = tomllib.load(f)
    except tomllib.TOMLDecodeError:
        print("Note: the old secrets.toml had formatting errors, so it was replaced.")

lines = [f"{k} = {json.dumps(str(v))}" for k, v in existing.items() if not isinstance(v, dict)]
lines += [
    "",
    "[auth]",
    f"redirect_uri = {json.dumps(REDIRECT)}",
    f"cookie_secret = {json.dumps(secrets.token_urlsafe(32))}",
    "",
    "[auth.google]",
    f"client_id = {json.dumps(client_id)}",
    f"client_secret = {json.dumps(client_secret)}",
    f"server_metadata_url = {json.dumps(METADATA)}",
    "",
]
os.makedirs(".streamlit", exist_ok=True)
with open(PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines).lstrip("\n"))

with open(PATH, "rb") as f:
    tomllib.load(f)  # sanity check: the file we wrote is valid
print(f"\nSaved {PATH}. Now run run.bat; the login page will show Continue with Google.")
