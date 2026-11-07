import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("GROQ_API_KEY", "test-key")  # modules check for it; no real calls are made
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
