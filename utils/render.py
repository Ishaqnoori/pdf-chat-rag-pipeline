"""
Renders an LLM answer in Streamlit: Markdown (headings, lists, tables) plus
```chart JSON blocks drawn as Altair charts.
"""
import json
import re
from functools import lru_cache

import altair as alt
import pandas as pd
import streamlit as st

from utils.security import sanitize_markdown

CHART_BLOCK = re.compile(r"```chart\s*\n(.*?)```", re.DOTALL)
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")

# Mid-tone colours that read well on both light and dark backgrounds
PALETTE = ["#3B82F6", "#F97316", "#10B981", "#8B5CF6", "#EAB308", "#EC4899"]


def _fix_tables(md: str) -> str:
    """Markdown tables only render if a blank line precedes them; models often forget."""
    lines, out = md.split("\n"), []
    for i, line in enumerate(lines):
        is_row = bool(TABLE_ROW.match(line))
        prev_is_row = i > 0 and bool(TABLE_ROW.match(lines[i - 1]))
        if is_row and not prev_is_row and out and out[-1].strip():
            out.append("")
        if not is_row and prev_is_row and line.strip():
            out.append("")
        out.append(line)
    return "\n".join(out)


def _to_number(v):
    if isinstance(v, (int, float)):
        return float(v)
    cleaned = re.sub(r"[^\d.\-]", "", str(v))
    return float(cleaned) if cleaned not in ("", "-", ".") else None


def _build_chart(spec: dict):
    kind = str(spec.get("type", "bar")).lower()
    labels = [str(x) for x in spec.get("labels", [])]
    series = spec.get("series") or []
    if not labels or not series:
        raise ValueError("chart needs labels and series")

    rows = []
    for s in series:
        name = str(s.get("name", "Value"))
        for label, val in zip(labels, s.get("values", [])):
            num = _to_number(val)
            if num is not None:
                rows.append({"label": label, "series": name, "value": num})
    if len(rows) < 2:
        raise ValueError("not enough numeric values")
    df = pd.DataFrame(rows)

    x_title = spec.get("x_label") or None
    y_title = spec.get("y_label") or None
    multi = df["series"].nunique() > 1
    color = alt.Color("series:N", title=None, scale=alt.Scale(range=PALETTE)) if multi else alt.value(PALETTE[0])
    x = alt.X("label:N", sort=labels, title=x_title, axis=alt.Axis(labelAngle=0, labelLimit=140))

    if kind == "pie":
        first = df[df["series"] == df["series"].iloc[0]]
        chart = alt.Chart(first).mark_arc(innerRadius=55, strokeWidth=0).encode(
            theta=alt.Theta("value:Q"),
            color=alt.Color("label:N", title=None, sort=labels, scale=alt.Scale(range=PALETTE)),
            tooltip=["label", alt.Tooltip("value:Q", format=",.2f")],
        )
    elif kind == "line":
        chart = alt.Chart(df).mark_line(point=True, strokeWidth=2.5).encode(
            x=x, y=alt.Y("value:Q", title=y_title), color=color,
            tooltip=["label", "series", alt.Tooltip("value:Q", format=",.2f")],
        )
    else:
        enc = dict(x=x, y=alt.Y("value:Q", title=y_title), color=color,
                   tooltip=["label", "series", alt.Tooltip("value:Q", format=",.2f")])
        if multi:
            enc["xOffset"] = "series:N"
        chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(**enc)

    title = spec.get("title")
    return (
        chart.properties(height=300, title=alt.Title(title, anchor="start") if title else "")
        .configure(font="Segoe UI, system-ui, sans-serif", background="transparent")
        .configure_view(strokeWidth=0)
        .configure_title(fontSize=14, fontWeight=600)
        .configure_legend(orient="bottom")
    )


@lru_cache(maxsize=256)
def _parse_answer(text: str) -> tuple:
    """
    Splits an answer into ("md", markdown) and ("chart", chart_or_None) parts. Cached: Streamlit
    re-runs on every click, and without this every past answer (and chart) would be rebuilt each time.
    """
    text = sanitize_markdown(text)
    parts, pos = [], 0
    for m in CHART_BLOCK.finditer(text):
        before = text[pos:m.start()].strip()
        if before:
            parts.append(("md", _fix_tables(before)))
        try:
            parts.append(("chart", _build_chart(json.loads(m.group(1)))))
        except Exception:
            parts.append(("chart", None))
        pos = m.end()
    rest = text[pos:].strip()
    if rest:
        parts.append(("md", _fix_tables(rest)))
    return tuple(parts)


def render_answer(text: str):
    """Writes the (sanitised) answer to the current Streamlit container."""
    for kind, part in _parse_answer(text):
        if kind == "md":
            st.markdown(part)
        elif part is not None:
            st.altair_chart(part, use_container_width=True)
        else:
            st.caption("A chart was suggested for this answer but its data couldn't be drawn.")


# ---------------------------------------------------------------- follow-ups & streaming
FOLLOWUP_BLOCK = re.compile(r"```followups\s*\n(.*?)(?:```|$)", re.DOTALL)
_HIDDEN_START = re.compile(r"```(?:chart|followups)")
_PARTIAL_FENCE = re.compile(r"`{1,3}[a-z]*$")  # a fence still arriving, e.g. "```follo"


def split_followups(text: str):
    """Returns (answer_without_block, [up to 3 follow-up questions])."""
    m = FOLLOWUP_BLOCK.search(text or "")
    if not m:
        return (text or "").strip(), []
    questions = [q.strip(" -*•\t").strip() for q in m.group(1).splitlines()]
    questions = [q for q in questions if 3 <= len(q) <= 120][:3]
    return (text[: m.start()] + text[m.end():]).strip(), questions


def stream_preview(text: str) -> str:
    """
    What to show while an answer is still arriving: everything before the first chart or
    follow-up block (half-written JSON would otherwise flash on screen).
    """
    m = _HIDDEN_START.search(text)
    visible = text[: m.start()] if m else _PARTIAL_FENCE.sub("", text)
    if m and text[m.start():].startswith("```chart"):
        visible += "\n\n*Drawing chart…*"
    return sanitize_markdown(visible)
