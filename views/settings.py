"""
Settings pages: Usage (overview and analytics), Account (export and delete), Privacy, Terms, and the
owner-only Admin page. `ctx` carries the signed-in user and the app callbacks these pages need.
"""
import html
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

from utils import auth, legal, quota, session, store
from utils.llm import AVAILABLE_MODELS, MODEL_IDS
from utils.security import duration_text, limiter
from utils.stats import is_admin, stats


@dataclass
class Ctx:
    user: dict
    user_key: str            # rate-limit identity
    store_uid: str           # Firestore document id
    save_chat: Callable      # saves the open chat into Recents before exporting


USAGE_CARDS = [  # (title, limit kind) on Settings → Usage → Overview
    ("Daily limit", "question_day"),
    ("Short-term limit (10 minutes)", "question"),
]


def visitor_timezone():
    """The browser's time zone (Streamlit >= 1.44), so charts show local hours even on a UTC server."""
    try:
        return ZoneInfo(st.context.timezone)
    except Exception:
        return timezone.utc


def usage_card(ctx, title: str, kind: str, key: str | None = None) -> str:
    u = limiter().usage(kind, key or ctx.user_key)
    left_pct = max(0, round(100 * (u["limit"] - u["used"]) / u["limit"]))
    reset = f"Resets in {duration_text(u['full_reset_in'])}" if u["used"] else "Full allowance available"
    level = "full" if left_pct == 0 else "low" if left_pct <= 20 else ""
    return (
        f'<div class="set-card"><div class="set-card-title">{title}</div>'
        f'<div class="set-row"><span>{reset}</span><span>{left_pct}% remaining</span></div>'
        f'<div class="set-bar {level}"><i style="width:{left_pct}%"></i></div></div>'
    )


def model_card(ctx, model: str) -> str:
    paused = quota.paused_for(model)
    if not paused:
        return usage_card(ctx, quota.model_name(model), f"model_day:{model}")
    return (  # paused by the provider for everyone: show it as empty until the reset
        f'<div class="set-card"><div class="set-card-title">{html.escape(quota.model_name(model))}</div>'
        f'<div class="set-row"><span>Usage limit reached. Resets in {duration_text(paused)}</span>'
        f'<span>0% remaining</span></div>'
        f'<div class="set-bar full"><i style="width:0%"></i></div></div>'
    )


def session_answers(ctx) -> list[dict]:
    """Every answer in this session's chats (current chat included, counted once)."""
    ss = st.session_state
    chats = {cid: c["messages"] for cid, c in ss.chats.items()}
    chats[ss.chat_id] = ss.messages
    return [m for msgs in chats.values() for m in msgs if m.get("kind") == "answer"]


def render_analytics(ctx):
    answers = session_answers(ctx)
    timed = [m["seconds"] for m in answers if m.get("seconds")]
    models = [AVAILABLE_MODELS.get(m["model"], (m["model"],))[0] for m in answers if m.get("model")]
    used_today = limiter().usage("question_day", ctx.user_key)["used"]
    tiles = [
        ("Questions today", str(used_today)),
        ("Answers this session", str(len(answers))),
        ("Average answer time", f"{sum(timed) / len(timed):.1f}s" if timed else "–"),
        ("Most used model", max(set(models), key=models.count) if models else "–"),
    ]
    st.markdown(
        '<div class="set-page"><h2>Your activity</h2><p>Based on today’s questions and this session’s chats.</p>'
        '<div class="set-tiles">'
        + "".join(f'<div class="set-tile"><span>{k}</span><b>{html.escape(v)}</b></div>' for k, v in tiles)
        + "</div></div>",
        unsafe_allow_html=True,
    )
    times = limiter().events("question_day", ctx.user_key)
    if times:
        now = time.time()
        hours_ago = [int((now - t) // 3600) for t in times]
        df = pd.DataFrame({"hours_ago": list(range(23, -1, -1))})
        df["Questions"] = [hours_ago.count(h) for h in df["hours_ago"]]
        tz = visitor_timezone()
        df["Hour"] = [datetime.fromtimestamp(now - h * 3600, tz).strftime("%H:00") for h in df["hours_ago"]]
        chart = (
            alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#3B82F6")
            .encode(x=alt.X("Hour:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelOverlap=True)),
                    y=alt.Y("Questions:Q", title=None, axis=alt.Axis(tickMinStep=1)),
                    tooltip=["Hour", "Questions"])
            .properties(height=220, title=alt.Title("Questions in the last 24 hours", anchor="start"))
            .configure(background="transparent").configure_view(strokeWidth=0)
            .configure_title(fontSize=14, fontWeight=600)
        )
        with st.container(key="set_chart"):
            st.altair_chart(chart, use_container_width=True)
    else:
        st.markdown('<div class="set-page"><p>No questions yet today. Your activity will show up here.</p></div>',
                    unsafe_allow_html=True)


def render_admin(ctx):
    """Owner-only overview of the whole app. Access is checked again here, on the server."""
    if not is_admin(ctx.user["email"]):
        st.session_state.settings_tab = "usage"
        return render(ctx)
    snap = stats().snapshot()
    saved_feedback = store.recent_feedback(20)  # survives restarts when the database is set up
    if saved_feedback is not None:
        snap["feedback"] = saved_feedback
    answers = snap["answers"]
    total = sum(a["count"] for a in answers.values())
    errors = sum(a["errors"] for a in answers.values())
    secs = sum(a["seconds"] for a in answers.values())
    ratings = [f["rating"] for f in snap["feedback"]]
    tiles = [
        ("Running for", duration_text(int(snap["uptime"]))),
        ("Answers", str(total)),
        ("Average answer time", f"{secs / total:.1f}s" if total else "–"),
        ("Errors", str(errors)),
        ("Helpful answers", f"{round(100 * sum(ratings) / len(ratings))}%" if ratings else "–"),
    ]
    app_cards = [("Questions today, all users", "app_question_day"), ("Uploads today, all users", "app_upload_day"),
                 ("Scanned PDFs today, all users", "app_ocr_day")]
    model_rows = ""
    for m in MODEL_IDS:
        a = answers.get(m, {})
        n = a.get("count", 0)
        avg = f"{a['seconds'] / n:.1f}s" if n else "–"
        model_rows += (f"<tr><td>{html.escape(quota.model_name(m))}</td><td>{n}</td><td>{avg}</td>"
                       f"<td>{a.get('errors', 0)}</td></tr>")
    st.markdown(
        '<div class="set-page"><h1>Admin</h1><p>Only visible to emails in <code>ADMIN_EMAILS</code>. '
        "Counters cover all users since the app last started.</p>"
        '<div class="set-tiles">'
        + "".join(f'<div class="set-tile"><span>{k}</span><b>{html.escape(v)}</b></div>' for k, v in tiles)
        + '</div><h2>App-wide limits</h2>'
        + "".join(usage_card(ctx, t, k, key="app") for t, k in app_cards)
        + "".join(model_card_app(ctx, m) for m in MODEL_IDS)
        + '<h2 class="set-h2-gap">Models</h2><table class="set-table"><tr><th>Model</th><th>Answers</th>'
        "<th>Avg time</th><th>Errors</th></tr>" + model_rows + "</table>"
        + '<h2 class="set-h2-gap">Recent feedback</h2>'
        + ("".join(
            f'<div class="set-fb"><span>{"👍" if f["rating"] else "👎"}</span>'
            f'<div><b>{html.escape(f["question"] or "(no question)")}</b>'
            f'<small>{html.escape(quota.model_name(f.get("model") or ""))}, {html.escape(f["time"])}</small>'
            f"</div></div>"
            for f in snap["feedback"][:20]) or "<p>No feedback yet.</p>")
        + "</div>",
        unsafe_allow_html=True,
    )
    with st.container(key="set_logout"):
        with st.expander("Sign everyone out"):
            st.markdown("Ends every saved \"keep me signed in\" login at once, for example after a secret "
                        "leaked. People who are using the app right now stay signed in until they log out.")
            if st.button("Sign everyone out", key="rotate_sessions", type="primary"):
                (st.success if session.sign_everyone_out() else st.error)(
                    "Done: all saved logins have ended." if store.enabled() else "Needs the Firestore database.")
    if snap["feedback"]:
        with st.container(key="set_logout"):
            lines = "\n".join(json.dumps(f, ensure_ascii=False) for f in snap["feedback"])
            st.download_button("Download feedback (JSONL)", lines,
                               file_name="feedback.jsonl", mime="application/json", icon=":material/download:")


def model_card_app(ctx, model: str) -> str:
    paused = quota.paused_for(model)
    title = f"{quota.model_name(model)}, all users"
    if not paused:
        return usage_card(ctx, title, f"app_model_day:{model}", key="app")
    return (f'<div class="set-card"><div class="set-card-title">{html.escape(title)}</div>'
            f'<div class="set-row"><span>Paused by the provider. Resets in {duration_text(paused)}</span>'
            f'<span>0% remaining</span></div><div class="set-bar full"><i style="width:0%"></i></div></div>')


def render_account_data(ctx):
    """Export my data and Delete account (privacy rights)."""
    st.markdown('<div class="set-page"><h2 class="set-h2-gap">Your data</h2>'
                "<p>Download everything saved to your account, or delete it permanently.</p></div>",
                unsafe_allow_html=True)
    with st.container(key="account_data"):
        # Reading every saved chat costs Firestore reads, so it only happens when asked
        if "export_data" not in st.session_state:
            if st.button("Prepare my data export", icon=":material/folder_zip:", key="prep_export"):
                ctx.save_chat()
                saved = store.export_user_chats(ctx.store_uid) if store.enabled() else []
                live = [{"id": cid, "title": c["title"], "updated": c["updated"],
                         "messages": [store._clean_message(m) for m in c["messages"]]}
                        for cid, c in st.session_state.chats.items()]
                st.session_state.export_data = json.dumps(
                    {"account": {"name": ctx.user["name"], "email": ctx.user["email"]}, "chats": saved or live},
                    ensure_ascii=False, indent=2)
                st.rerun()
        else:
            st.download_button("Download my data (JSON)", st.session_state.export_data,
                               file_name="ask-your-pdf-data.json", mime="application/json",
                               icon=":material/download:", on_click=lambda: st.session_state.pop("export_data", None))
        with st.expander("Delete account"):
            st.markdown("This permanently deletes your account, saved chats, sessions and feedback. "
                        "It can't be undone.")
            if ctx.user.get("method") == "firebase":
                confirm = st.text_input("Password", type="password", key="delete_pw", placeholder="Your password")
            else:
                st.markdown("Your Google account itself isn't affected; only your data in this app is deleted.")
                typed = st.text_input("Type DELETE to confirm", key="delete_confirm", placeholder="DELETE")
                confirm = "ok" if typed.strip() == "DELETE" else None
            if st.button("Delete my account", key="delete_account_btn", type="primary"):
                ok, message = auth.delete_account(confirm if ctx.user.get("method") == "firebase" else None) \
                    if confirm else (False, "Confirm first, as described above.")
                if ok:
                    st.rerun()
                st.error(message)


def render(ctx):
    tab = st.session_state.settings_tab
    if tab in ("privacy", "terms"):
        st.markdown('<div class="set-page"></div>', unsafe_allow_html=True)
        with st.container(key="legal_page"):
            st.markdown(legal.privacy() if tab == "privacy" else legal.terms())
        return
    if tab == "admin":
        return render_admin(ctx)
    if tab == "account":
        methods = {"google": "Google", "firebase": "Email and password"}
        method = methods.get(ctx.user.get("method"), "Email and password")
        rows = [("Name", ctx.user["name"]), ("Email", ctx.user["email"]), ("Signed in with", method)]
        st.markdown(
            '<div class="set-page"><h1>Account</h1><div class="set-card">'
            + "".join(f'<div class="set-kv"><span>{k}</span><span>{html.escape(v)}</span></div>' for k, v in rows)
            + "</div></div>",
            unsafe_allow_html=True,
        )
        with st.container(key="set_logout"):
            if st.button("Log out", icon=":material/logout:", key="set_logout_btn"):
                auth.logout()
                st.rerun()
        render_account_data(ctx)
        return
    st.markdown('<div class="set-page"><h1>Usage</h1></div>', unsafe_allow_html=True)
    with st.container(key="usage_tabs"):
        view = st.segmented_control("View", ["Overview", "Analytics"], default="Overview",
                                    key="usage_view", label_visibility="collapsed") or "Overview"
    if view == "Analytics":
        return render_analytics(ctx)
    lock = quota.user_lock(ctx.user_key)
    st.markdown(
        '<div class="set-page"><h2>Question limits</h2>'
        "<p>Shared across all your chats and PDFs. These limits keep the app’s AI quota from running out "
        "for everyone; each one refills gradually after you use it.</p>"
        + "".join(usage_card(ctx, t, k) for t, k in USAGE_CARDS)
        + "<h2 class=\"set-h2-gap\">Model limits</h2>"
        "<p>Each model has its own daily allowance. If you use up any one of them, all models are paused "
        "for you until it resets, so the limits can't be bypassed by switching models.</p>"
        + (f'<div class="set-alert">{html.escape(quota.lock_message(lock))}</div>' if lock else "")
        + "".join(model_card(ctx, m) for m in MODEL_IDS)
        + "</div>",
        unsafe_allow_html=True,
    )


def sidebar(ctx):
    with st.container(key="settings_nav"):
        if st.button("Back to app", icon=":material/arrow_back:", key="set_back", use_container_width=True):
            st.session_state.view = "chat"
            st.rerun()
        st.markdown('<p class="side-label">Settings</p>', unsafe_allow_html=True)
        tabs = [("usage", "Usage", ":material/speed:"), ("account", "Account", ":material/person:"),
                ("privacy", "Privacy", ":material/shield:"), ("terms", "Terms", ":material/description:")]
        if is_admin(ctx.user["email"]):
            tabs.append(("admin", "Admin", ":material/admin_panel_settings:"))
        for tab, label, icon in tabs:
            if st.button(label, icon=icon, key=f"set_{tab}", use_container_width=True):
                st.session_state.settings_tab = tab
                st.rerun()
