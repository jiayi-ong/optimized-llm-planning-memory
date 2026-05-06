"""
app/utils/ui_style.py
======================
Shared Streamlit CSS injection for the eval developer UI.

Color palette: orange-red accent on dark neutral background.
  Primary   #E84545  — alert red-orange (CTAs, key metric highlights)
  Secondary #FF6B35  — warm orange (secondary accents, charts)
  Neutral   #1E1E2E  — near-black background
  Surface   #2A2A3E  — card / table background
  Text      #E0E0E0  — body text
  Muted     #9090A8  — captions, secondary text

Usage
-----
    from app.utils.ui_style import inject_css, metric_card, scrollable_container
    inject_css()
"""

from __future__ import annotations

import streamlit as st

# ── Palette constants (use in Python charts too) ──────────────────────────────
PRIMARY = "#E84545"
SECONDARY = "#FF6B35"
ACCENT_LIGHT = "#FF9B71"
SURFACE = "#2A2A3E"
TEXT = "#E0E0E0"
MUTED = "#9090A8"
SUCCESS = "#4CAF50"
WARNING = "#FFC107"

# For Plotly / matplotlib charts
PALETTE = [PRIMARY, SECONDARY, ACCENT_LIGHT, "#4A90D9", "#A78BFA", "#34D399"]

CSS = """
<style>
/* ── Base ─────────────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    font-size: 15px;
    color: #E0E0E0;
}
[data-testid="stSidebar"] { background: #1A1A2E; }
[data-testid="stSidebar"] label { font-size: 13px; color: #C0C0D8; }

/* ── Headings ─────────────────────────────────────────────────────────────── */
h1 { font-size: 1.8rem !important; color: #E84545 !important; font-weight: 700; }
h2 { font-size: 1.3rem !important; color: #FF6B35 !important; }
h3 { font-size: 1.05rem !important; color: #E0E0E0 !important; }

/* ── Metric widgets ───────────────────────────────────────────────────────── */
[data-testid="stMetric"] { background: #2A2A3E; border-radius: 8px; padding: 0.6rem 1rem; }
[data-testid="stMetricLabel"] { font-size: 12px !important; color: #9090A8 !important; }
[data-testid="stMetricValue"] { font-size: 22px !important; color: #FF6B35 !important; font-weight: 700; }
[data-testid="stMetricDelta"] { font-size: 13px !important; }

/* ── Buttons ─────────────────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    background: #E84545;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    font-size: 14px;
    padding: 0.35rem 1rem;
}
[data-testid="stButton"] > button:hover { background: #C62A2A; }

/* ── Dataframes / tables ─────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { font-size: 13px; }
thead tr th { background: #2A2A3E !important; color: #FF6B35 !important; font-size: 12px; }
tbody tr:hover { background: #33334A !important; }

/* ── Expanders ────────────────────────────────────────────────────────────── */
details summary { font-size: 14px; color: #FF6B35; }
details[open] summary { color: #E84545; }

/* ── Captions ─────────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] { font-size: 12px !important; color: #9090A8 !important; }

/* ── Selectbox / multiselect ──────────────────────────────────────────────── */
[data-baseweb="select"] { font-size: 13px; }

/* ── Scrollable block wrapper ─────────────────────────────────────────────── */
.scroll-block {
    max-height: 420px;
    overflow-y: auto;
    border: 1px solid #3A3A5A;
    border-radius: 6px;
    padding: 0.5rem;
    background: #1E1E2E;
}

/* ── Status badges ────────────────────────────────────────────────────────── */
.badge-running  { background:#FF6B35; color:#fff; padding:2px 8px; border-radius:4px; font-size:11px; }
.badge-completed{ background:#4CAF50; color:#fff; padding:2px 8px; border-radius:4px; font-size:11px; }
.badge-failed   { background:#E84545; color:#fff; padding:2px 8px; border-radius:4px; font-size:11px; }
.badge-cancelled{ background:#9090A8; color:#fff; padding:2px 8px; border-radius:4px; font-size:11px; }
</style>
"""


def inject_css() -> None:
    """Inject the shared CSS block into the current Streamlit page."""
    st.markdown(CSS, unsafe_allow_html=True)


def score_color(score: float) -> str:
    """Return a hex color for a [0,1] score (red → yellow → green)."""
    if score >= 0.75:
        return SUCCESS
    if score >= 0.45:
        return WARNING
    return PRIMARY


def badge(status: str) -> str:
    """Return an HTML badge span for a job/run status string."""
    cls = f"badge-{status.lower()}"
    return f'<span class="{cls}">{status.upper()}</span>'


def metric_card(label: str, value: str, delta: str | None = None) -> None:
    """Render a styled metric inside its own card column."""
    delta_html = f"<div style='font-size:12px;color:#9090A8'>{delta}</div>" if delta else ""
    st.markdown(
        f"""
        <div style='background:#2A2A3E;border-radius:8px;padding:0.7rem 1rem;margin-bottom:0.5rem'>
          <div style='font-size:12px;color:#9090A8'>{label}</div>
          <div style='font-size:22px;font-weight:700;color:#FF6B35'>{value}</div>
          {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
