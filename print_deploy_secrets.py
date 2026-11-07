"""
Prints the [firebase_service_account] block to paste into Streamlit Cloud > App settings > Secrets.
Run:  venv\\Scripts\\python print_deploy_secrets.py
The output contains your private key: paste it only into the host's Secrets box, nowhere else.
"""
import json
import os

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "firebase-service-account.json")
with open(path, encoding="utf-8") as f:
    info = json.load(f)

print("[firebase_service_account]")
for key, value in info.items():
    print(f"{key} = {json.dumps(value)}")  # JSON string escaping is valid TOML
