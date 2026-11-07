"""Login, sign-up, password reset, and the public Privacy / Terms pages (shown before signing in)."""
import time

import streamlit as st

from utils import auth, legal, session, store
from utils.brand import logo_svg


def render():
    st.markdown(
        "<style>[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'],"
        "[data-testid='stExpandSidebarButton'] { display: none !important; }</style>",
        unsafe_allow_html=True,
    )
    if st.session_state.pop("clear_cookie", False):
        session.write_cookie(None)  # finish logging out: remove the "keep me signed in" cookie
    view = st.session_state.get("auth_view", "login")
    if view in ("privacy", "terms"):
        with st.container(key="legal_page"):
            st.markdown(legal.privacy() if view == "privacy" else legal.terms())
            if st.button("Back", icon=":material/arrow_back:", key="legal_back"):
                st.session_state.auth_view = "login"
                st.rerun()
        return
    title, sub = {
        "signup": ("Create your account", "Sign up to start chatting with your documents."),
        "reset": ("Reset your password", "We'll email you a link to choose a new one."),
    }.get(view, ("Welcome back", "Log in to chat with your documents."))
    st.markdown(
        f'<div class="login"><div class="brand-lg">{logo_svg(44)}<span>Ask your PDF</span></div>'
        f'<h1>{title}</h1><p>{sub}</p></div>',
        unsafe_allow_html=True,
    )

    def switch(label, to, key):
        with st.container(key=key):
            if st.button(label, use_container_width=True):
                st.session_state.auth_view = to
                st.rerun()

    with st.container(key="login_box"):
        notice = st.session_state.pop("auth_notice", None)
        if notice:
            st.info(notice)

        if view != "reset" and auth.google_configured():
            with st.container(key="google_login"):
                if st.button("Continue with Google", use_container_width=True):
                    st.login("google")
            st.markdown('<div class="or">or</div>', unsafe_allow_html=True)

        if view == "signup":
            st.session_state.setdefault("signup_shown_at", time.time())
            with st.form("signup_form", border=False):
                name = st.text_input("Name", placeholder="Full name", max_chars=80, label_visibility="collapsed")
                email = st.text_input("Email", placeholder="Email address", max_chars=254, label_visibility="collapsed")
                password = st.text_input("Password", type="password", placeholder="Password (8+ characters)",
                                         max_chars=128, label_visibility="collapsed")
                confirm = st.text_input("Confirm", type="password", placeholder="Confirm password",
                                        max_chars=128, label_visibility="collapsed")
                with st.container(key="hp"):  # hidden from people by CSS; bots fill it in
                    honeypot = st.text_input("Website", key="hp_website", label_visibility="collapsed")
                agreed = st.checkbox("I agree to the Terms of Service and Privacy Policy", key="agree_terms")
                submitted = st.form_submit_button("Create account", use_container_width=True)
            if submitted:
                with st.spinner("Creating your account…"):
                    ok, msg = auth.sign_up(name, email, password, confirm, agreed=agreed, honeypot=honeypot,
                                           form_shown_at=st.session_state.get("signup_shown_at"))
                if ok:
                    st.session_state.pop("signup_shown_at", None)
                    st.session_state.auth_view = "login"
                    st.session_state.auth_notice = msg
                    st.rerun()
                st.error(msg)
            switch("Already have an account? Log in", "login", "auth_switch")

        elif view == "reset":
            with st.form("reset_form", border=False):
                email = st.text_input("Email", placeholder="Email address", max_chars=254, label_visibility="collapsed")
                submitted = st.form_submit_button("Send reset link", use_container_width=True)
            if submitted:
                ok, msg = auth.send_reset(email)
                (st.success if ok else st.error)(msg)
            switch("Back to log in", "login", "auth_switch")

        else:
            with st.form("login_form", border=False):
                email = st.text_input("Email", placeholder="Email", max_chars=254, label_visibility="collapsed")
                password = st.text_input("Password", type="password", placeholder="Password",
                                         max_chars=128, label_visibility="collapsed")
                remember = False
                if session.LIFETIME > 0 and store.enabled():
                    days = session.LIFETIME // 86400
                    remember = st.checkbox(f"Keep me signed in on this browser for {days} days", value=False,
                                           key="remember_me")
                submitted = st.form_submit_button("Log in", use_container_width=True)
            if submitted:
                with st.spinner("Logging in…"):
                    ok, msg = auth.login(email, password, remember=remember)
                if ok:
                    st.rerun()
                st.error(msg)
            switch("Forgot password?", "reset", "auth_forgot")
            switch("Don't have an account? Sign up", "signup", "auth_switch")

    st.markdown('<p class="legal">PDFs are never stored. Your chats are saved to your account '
                'so you can come back to them.</p>', unsafe_allow_html=True)
    with st.container(key="legal_links", horizontal=True):
        for label, target in (("Privacy", "privacy"), ("Terms", "terms")):
            if st.button(label, key=f"open_{target}"):
                st.session_state.auth_view = target
                st.rerun()
