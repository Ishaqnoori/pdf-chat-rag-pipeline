"""
Error alerts with Sentry's free plan: crashes and logged errors are emailed to you.
Only active when SENTRY_DSN is set. Personal data is never sent: no IPs, cookies, request
bodies or local variables (which could contain questions or document text).
"""
import logging
import os

import streamlit as st


def _scrub(event, hint):
    event.pop("request", None)
    event.pop("user", None)
    for exc in (event.get("exception") or {}).get("values", []):
        for frame in (exc.get("stacktrace") or {}).get("frames", []):
            frame.pop("vars", None)
    return event


@st.cache_resource(show_spinner=False)
def init() -> bool:
    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
        return False
    try:
        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration

        sentry_sdk.init(
            dsn=dsn,
            send_default_pii=False,
            include_local_variables=False,
            traces_sample_rate=0.0,  # errors only: keeps you inside the free plan
            before_send=_scrub,
            integrations=[LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)],
        )
        return True
    except Exception:
        logging.getLogger("askpdf").exception("Sentry setup failed")
        return False
