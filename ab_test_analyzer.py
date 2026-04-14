import streamlit as st
import numpy as np
import json
import io
import datetime
import re as _re
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # Must be set before importing pyplot
import matplotlib.pyplot as plt

# ── Global chart theme ───────────────────────────────────────────────────────
# Match Streamlit's dark UI. Applied once at module level — all plot functions inherit.
_BG      = "#0e1117"   # Streamlit dark background
_BG_AX   = "#1a1d24"   # Slightly lighter for axes area
_FG      = "#e0e0e0"   # Primary text / tick labels
_GRID    = "#2e3140"   # Subtle grid lines
_SPINE   = "#3a3f52"   # Axis border colour

plt.rcParams.update({
    # Figure & axes backgrounds
    "figure.facecolor":     _BG,
    "axes.facecolor":       _BG_AX,
    "savefig.facecolor":    _BG,
    # Text & labels
    "text.color":           _FG,
    "axes.labelcolor":      _FG,
    "axes.titlecolor":      _FG,
    "axes.titlesize":       13,
    "axes.labelsize":       11,
    "axes.titlepad":        14,
    # Ticks
    "xtick.color":          _FG,
    "ytick.color":          _FG,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    # Grid
    "axes.grid":            True,
    "grid.color":           _GRID,
    "grid.linewidth":       0.7,
    "grid.alpha":           1.0,
    # Spines
    "axes.edgecolor":       _SPINE,
    "axes.linewidth":       0.8,
    # Legend
    "legend.facecolor":     _BG_AX,
    "legend.edgecolor":     _SPINE,
    "legend.labelcolor":    _FG,
    "legend.fontsize":      10,
    # Layout
    "figure.dpi":           120,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
})
from scipy.stats import beta, chisquare, mannwhitneyu, chi2_contingency
from statsmodels.stats.proportion import proportions_ztest, proportion_confint, proportion_effectsize
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.multitest import multipletests
import openai
import plotly.graph_objects as go
import plotly.figure_factory as ff

# -----------------------------------------------
# JSON HELPERS
# -----------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)

# -----------------------------------------------
# PAGE CONFIG
# -----------------------------------------------
st.set_page_config(page_title="Enterprise A/B Test Analyzer", layout="wide", initial_sidebar_state="expanded")

# -----------------------------------------------
# SVG ICONS
# -----------------------------------------------
ICON_SETTINGS  = """<svg viewBox="0 0 1024 1024" style="width:1.5em;height:1.5em;vertical-align:middle;margin-right:10px;" xmlns="http://www.w3.org/2000/svg"><path d="M512 512m-10 0a10 10 0 1 0 20 0 10 10 0 1 0-20 0Z" fill="#E73B37"/><path d="M512 306.8c27.7 0 54.6 5.4 79.8 16.1 24.4 10.3 46.4 25.1 65.2 44s33.6 40.8 44 65.2c10.7 25.3 16.1 52.1 16.1 79.8 0 27.7-5.4 54.6-16.1 79.8-10.3 24.4-25.1 46.4-44 65.2-18.8 18.8-40.8 33.6-65.2 44-25.3 10.7-52.1 16.1-79.8 16.1-27.7 0-54.6-5.4-79.8-16.1-24.4-10.3-46.4-25.1-65.2-44-18.8-18.8-33.6-40.8-44-65.2-10.7-25.3-16.1-52.1-16.1-79.8 0-27.7 5.4-54.6 16.1-79.8 10.3-24.4 25.1-46.4 44-65.2s40.8-33.6 65.2-44c25.2-10.6 52.1-16.1 79.8-16.1m0-22c-125.4 0-227.1 101.7-227.1 227.1S386.6 739.1 512 739.1c125.4 0 227.1-101.7 227.1-227.1S637.4 284.8 512 284.8z" fill="#ffffff"/><path d="M544.2 107.3l34.1 92.3 7.4 19.9 20.2 6.6c10.3 3.4 32.1 12.9 43.4 18.1l18.7 8.6 18.6-8.9 87.9-41.8 46.4 46.5-41.2 89.4-8.9 19.3 9.6 19c6.8 13.4 12.6 27.5 17.4 41.9l6.7 20.5 20.3 7.2 91.7 32.6v65.7l-92.3 34.1-19.9 7.4-6.6 20.2c-4.7 14.4-10.6 28.4-17.4 41.9l-9.8 19.3 9.3 19.5 41.8 87.9-46.5 46.5-89.1-41.3-19.3-8.9-19 9.6c-13.4 6.8-27.5 12.6-41.9 17.4l-20.5 6.7-7.2 20.3-32.6 91.7h-65.7l-34.1-92.3-7.4-19.9-20.2-6.6c-10.3-3.4-32.1-12.9-43.4-18.1L356 771l-18.6 8.9-87.9 41.8-46.4-46.5 41.2-89.3 8.9-19.3-9.6-19c-6.8-13.4-12.6-27.5-17.4-41.9l-6.7-20.5-20.3-7.2-91.7-32.6v-65.7l92.3-34.1 19.9-7.4 6.6-20.2c3.4-10.3 12.9-32.1 18.1-43.4l8.6-18.7-8.9-18.6-41.8-87.9 46.4-46.4 89.3 41.2 19.3 8.9 19-9.6c13.4-6.8 27.5-12.6 41.9-17.4l20.5-6.7 7.2-20.3 32.6-91.7h65.7m30.7-44.1H447.4l-43 121c-16.6 5.5-32.7 12.1-48.1 19.9l-117.2-54-90.1 90.1 55.2 116s-14.5 31.4-19.9 48.1l-121 44.7v127.4l121 43c5.5 16.6 12.1 32.6 19.9 48l-54 117.2 90.1 90.1 116-55.2s31.4 14.5 48.1 19.9l44.7 121h127.4l43-121c16.6-5.5 32.6-12.1 48-19.9l117.2 54 90.1-90.1-55.2-116c7.8-15.4 14.5-31.4 19.9-48l121-44.7V447.4l-121-43c-5.5-16.6-12.1-32.6-19.9-48l54-117.2-90.1-90.1-115.9 55.2s-31.5-14.5-48.1-19.9L574.9 63.3z" fill="#ffffff"/></svg>"""
ICON_CALENDAR  = """<svg viewBox="0 0 1024 1024" style="width:1.5em;height:1.5em;vertical-align:middle;margin-right:10px;" xmlns="http://www.w3.org/2000/svg"><path d="M716 190.9v-67.8h-44v67.8H352v-67.8h-44v67.8H92v710h840v-710H716z m-580 44h172v69.2h44v-69.2h320v69.2h44v-69.2h172v151.3H136V234.9z m752 622H136V402.2h752v454.7z" fill="#ffffff"/><path d="M319 565.7m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"/><path d="M510 565.7m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"/><path d="M701.1 565.7m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"/><path d="M319 693.4m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"/><path d="M510 693.4m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"/><path d="M701.1 693.4m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"/></svg>"""
ICON_BAR_CHART = """<svg viewBox="0 0 1024 1024" style="width:1.5em;height:1.5em;vertical-align:middle;margin-right:10px;" xmlns="http://www.w3.org/2000/svg"><path d="M928.1 881v44H95.9V99h44v782z" fill="#ffffff"/><path d="M352 435.7v403.4H204V435.7h148m22-22H182v447.4h192V413.7zM608 307.9v531.2H460V307.9h148m22-22H438v575.2h192V285.9z" fill="#ffffff"/><path d="M866.1 177.3v663.9H714V177.3h152.1m20-20H694v703.9h192V157.3h0.1z" fill="#E73B37"/></svg>"""
ICON_PIE       = """<svg viewBox="0 0 1024 1024" style="width:1.5em;height:1.5em;vertical-align:middle;margin-right:10px;" xmlns="http://www.w3.org/2000/svg"><path d="M429.9 186.7v406.4h407.5c-4 34.1-12.8 67.3-26.2 99.1-18.4 43.6-44.8 82.7-78.5 116.3-33.6 33.6-72.8 60-116.4 78.4-45.1 19.1-93 28.7-142.5 28.7-49.4 0-97.4-9.7-142.5-28.7-43.6-18.4-82.7-44.8-116.4-78.4-33.6-33.6-60-72.7-78.4-116.3-19.1-45.1-28.7-93-28.7-142.4s9.7-97.3 28.7-142.4c18.4-43.6 44.8-82.7 78.4-116.3 33.6-33.6 72.8-60 116.4-78.4 31.7-13.2 64.7-21.9 98.6-26m44-46.6c-226.4 0-410 183.5-410 409.8s183.6 409.8 410 409.8 410-183.5 410-409.8v-0.8h-410v-409z" fill="#ffffff"/><path d="M566.1 80.5c43.7 1.7 86.4 10.6 127 26.4 44 17.1 84.2 41.8 119.6 73.5 71.7 64.1 117.4 151.7 128.7 246.7 1.2 9.9 2 20 2.4 30.2H566.1V80.5m-16-16.3v409h410c0-16.3-1-32.3-2.9-48.1C933.1 221.9 760 64.2 550.1 64.2zM264.7 770.4c-23.1-23.1-42.3-49.1-57.3-77.7l-14.7 6.5c35.7 68.2 94 122.7 165 153.5l4.3-15.6c-36.3-16-69.1-38.4-97.3-66.7z" fill="#E73B37"/></svg>"""
ICON_BRAIN     = """<svg viewBox="0 0 1024 1024" style="width:1.5em;height:1.5em;vertical-align:middle;margin-right:10px;" xmlns="http://www.w3.org/2000/svg"><path d="M512 301.2m-10 0a10 10 0 1 0 20 0 10 10 0 1 0-20 0Z" fill="#E73B37"/><path d="M511.8 256.6c24.4 0 44.2 19.8 44.2 44.2S536.2 345 511.8 345s-44.2-19.8-44.2-44.2 19.9-44.2 44.2-44.2m0-20c-35.5 0-64.2 28.7-64.2 64.2s28.7 64.2 64.2 64.2 64.2-28.7 64.2-64.2-28.7-64.2-64.2-64.2z" fill="#E73B37"/><path d="M730.7 529.5c0.4-8.7 0.6-17.4 0.6-26.2 0-179.6-86.1-339.1-219.3-439.5-133.1 100.4-219.2 259.9-219.2 439.5 0 8.8 0.2 17.5 0.6 26.1-56 56-90.6 133.3-90.6 218.7 0 61.7 18 119.1 49.1 167.3 30.3-49.8 74.7-90.1 127.7-115.3 39-18.6 82.7-29 128.8-29 48.3 0 93.9 11.4 134.3 31.7 52.5 26.3 96.3 67.7 125.6 118.4 33.4-49.4 52.9-108.9 52.9-173.1 0-85.4-34.6-162.6-90.5-218.6z" fill="#ffffff"/><path d="M512 819.3c8.7 0 24.7 22.9 24.7 60.4s-16 60.4-24.7 60.4-24.7-22.9-24.7-60.4 16-60.4 24.7-60.4m0-20c-24.7 0-44.7 36-44.7 80.4 0 44.4 20 80.4 44.7 80.4s44.7-36 44.7-80.4c0-44.4-20-80.4-44.7-80.4z" fill="#E73B37"/></svg>"""
ICON_UPLOAD    = """<svg viewBox="0 0 1024 1024" style="width:1.5em;height:1.5em;vertical-align:middle;margin-right:10px;" xmlns="http://www.w3.org/2000/svg"><path d="M220.5 245.4c-32.8 32.8-55.1 73.2-65.2 117.3h16.5c18.8-75.3 75.1-135.9 148-160.7v-16.9c-37.1 11.6-71 32-99.3 60.3z" fill="#E73B37"/><path d="M959.9 540.8c0 113.6-92.1 205.8-205.7 205.9H590.9v-44h163.3c43.2 0 83.8-16.9 114.3-47.4 30.6-30.6 47.4-71.2 47.4-114.5 0-43.2-16.8-83.9-47.4-114.4S797.2 379 754 379c-11.5 0-22.8 1.2-33.8 3.5-15 3.2-29.4 8.4-42.8 15.7-1-15.4-3.3-30.7-6.8-45.6-3.6-15.6-8.6-30.8-14.9-45.7-14.4-33.9-34.9-64.4-61.1-90.6-26.2-26.2-56.6-46.7-90.6-61.1-35.1-14.8-72.4-22.4-110.9-22.4s-75.8 7.5-110.9 22.4c-33.9 14.3-64.4 34.9-90.6 61.1-26.2 26.2-46.7 56.7-61.1 90.6-14.9 35.1-22.4 72.4-22.4 110.9s7.5 75.8 22.4 110.9c14.3 33.9 34.9 64.4 61.1 90.6 26.2 26.2 56.7 46.7 90.6 61.1 35.1 14.8 72.4 22.4 110.9 22.4h39.7v44h-41C210.7 746 64.1 599 64.1 417.7c0-181.7 147.3-329 329-329 154.6 0 284.3 106.6 319.5 250.3 13.4-2.7 27.2-4.2 41.4-4.2 113.7 0.1 205.9 92.2 205.9 205.9z" fill="#ffffff"/><path d="M692.9 636.1h-22.6L519.8 485.6v449.6h-16V485.8L353.4 636.1h-22.6l181-181z" fill="#E73B37"/></svg>"""
ICON_GUARDRAIL = """<svg viewBox="0 0 1024 1024" style="width:1.5em;height:1.5em;vertical-align:middle;margin-right:10px;" xmlns="http://www.w3.org/2000/svg"><path d="M512 64L160 192v256c0 224 152 416 352 512 200-96 352-288 352-512V192L512 64z m308 384c0 192-128 364-308 452-180-88-308-260-308-452V228l308-112 308 112v220z" fill="#ffffff"/><path d="M512 320m-10 0a10 10 0 1 0 20 0 10 10 0 1 0-20 0Z" fill="#E73B37"/><path d="M512 276c24.4 0 44 19.6 44 44s-19.6 44-44 44-44-19.6-44-44 19.6-44 44-44m0-20c-35.3 0-64 28.7-64 64s28.7 64 64 64 64-28.7 64-64-28.7-64-64-64z" fill="#E73B37"/><path d="M420 550l60 60 140-140-28-28-112 112-32-32z" fill="#E73B37"/></svg>"""
ICON_TROPHY    = """<svg viewBox="0 0 1024 1024" style="width:1.5em;height:1.5em;vertical-align:middle;margin-right:10px;" xmlns="http://www.w3.org/2000/svg"><path d="M828.5 180.1h-9.9v-54.7h23.5v-44H182v44h23v54.7h-9.5C123.2 180.1 64 239.2 64 311.5v0.1c0 72.3 59.2 131.5 131.5 131.5h9.6c0 1.3 0.1 2.5 0.1 3.7 0.5 17.7 2.7 35.4 6.2 52.5 17.8 85.7 71.8 160 148.3 204 4.8 2.8 9.8 5.4 14.7 7.9 15.3 7.7 31.2 14.1 47.4 19.2 3.4 1 6.8 2 10.2 2.9v165.2H250.4v44h511.9v-44H591.9V733.4c3.7-1 7.3-2.1 10.9-3.2 16.2-5.1 32.2-11.6 47.4-19.4 5-2.5 10-5.3 14.8-8.1 75.6-43.9 129.2-117.8 147-202.7 3.6-17.2 5.8-34.9 6.3-52.4 0.1-1.5 0.1-3 0.1-4.5h10c72.3 0 131.5-59.2 131.5-131.5v-0.1c0.1-72.3-59.1-131.4-131.4-131.4zM205 399.2h-9.5c-23.2 0-45.1-9.1-61.7-25.7s-25.7-38.5-25.7-61.7v-0.1c0-23.2 9.1-45.2 25.7-61.7 16.6-16.6 38.5-25.7 61.7-25.7h9.5v174.9z m370.9 499.4h-128V737.3c20.9 4.5 42.3 6.8 63.9 6.8 21.7 0 43.1-2.3 64.1-6.8v161.3z m198.7-461.4c0 2.9 0 5.9-0.2 8.9-0.5 15-2.3 30.1-5.4 44.9-15.3 72.7-61.2 136-126.1 173.7-4.1 2.4-8.4 4.7-12.7 6.9-13 6.6-26.7 12.2-40.6 16.6-25.2 7.9-51.4 11.9-77.9 11.9-26.2 0-52.2-3.9-77.1-11.6-13.9-4.3-27.5-9.8-40.6-16.4-4.2-2.1-8.5-4.4-12.6-6.8-65.4-37.8-111.7-101.5-126.9-174.8-3.1-14.7-4.9-29.8-5.3-45-0.1-2.7-0.1-5.5-0.1-8.2v-312h525.6v311.9zM916 311.7c0 23.2-9.1 45.2-25.7 61.7-16.6 16.6-38.5 25.7-61.7 25.7h-9.9v-175h9.9c23.2 0 45.1 9.1 61.7 25.7s25.7 38.5 25.7 61.7v0.2z" fill="#ffffff"/><path d="M555.4 659.6l-4.8-19.4c0.3-0.1 26.5-6.8 55.4-23.5 37.8-21.9 62-49.7 72-82.7l19.1 5.8c-11.4 37.6-39.6 70.3-81.6 94.5-31.2 18-58.9 25-60.1 25.3z" fill="#E73B37"/></svg>"""

# -----------------------------------------------
# CONSTANTS
# -----------------------------------------------
BAYESIAN_SAMPLES  = 50_000
BOOTSTRAP_SAMPLES = 10_000
MAX_VARIATIONS    = 3   # max variation groups (so max total groups = 4: control + 3)
# One colour per group: Control, Var A, Var B, Var C
GROUP_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]


# -----------------------------------------------
# STATE INITIALIZATION
# -----------------------------------------------
def initialize_state():
    defaults = {
        "num_variations": 1,
        # Control
        "users_c": 5000, "conv_c": 500, "rev_c": 25000.0, "prod_c": 750,
        # Variation A
        "users_v0": 5000, "conv_v0": 600, "rev_v0": 33000.0, "prod_v0": 1000,
        # Variation B
        "users_v1": 5000, "conv_v1": 560, "rev_v1": 30000.0, "prod_v1": 900,
        # Variation C
        "users_v2": 5000, "conv_v2": 540, "rev_v2": 28000.0, "prod_v2": 850,
        "days": 14, "conf_level": "95%", "mc_method": "holm", "primary_goal": "Maximize CR",
        "start_date": None,
        "p_traffic": 50000, "p_base_cr": 2.5, "p_base_aov": 75.0,
        "p_mde": 5.0, "p_vol": "Medium (Standard E-com)",
        "s1_uc": 2000, "s1_cc": 100, "s1_uv": 2000, "s1_cv": 110,
        "s2_uc": 3000, "s2_cc": 400, "s2_uv": 3000, "s2_cv": 420,
        # Guardrail metrics
        "num_guardrails": 1,
        "g0_name": "Bounce Rate",       "g0_ctrl": 45.0, "g0_var": 0.0, "g0_threshold": 5.0, "g0_dir": "lower is better",
        "g1_name": "Session Duration",  "g1_ctrl": 0.0,  "g1_var": 0.0, "g1_threshold": 5.0, "g1_dir": "higher is better",
        "g2_name": "Add-to-Cart Rate",  "g2_ctrl": 0.0,  "g2_var": 0.0, "g2_threshold": 5.0, "g2_dir": "higher is better",
        # Segment breakdown
        "num_segments": 1,
        "seg0_name": "Mobile",   "seg1_name": "Desktop",   "seg2_name": "Segment 3", "seg3_name": "Segment 4",
        "seg0_uc": 0, "seg0_cc": 0, "seg0_rc": 0.0, "seg0_uv": 0, "seg0_cv": 0, "seg0_rv": 0.0,
        "seg1_uc": 0, "seg1_cc": 0, "seg1_rc": 0.0, "seg1_uv": 0, "seg1_cv": 0, "seg1_rv": 0.0,
        "seg2_uc": 0, "seg2_cc": 0, "seg2_rc": 0.0, "seg2_uv": 0, "seg2_cv": 0, "seg2_rv": 0.0,
        "seg3_uc": 0, "seg3_cc": 0, "seg3_rc": 0.0, "seg3_uv": 0, "seg3_cv": 0, "seg3_rv": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

initialize_state()

# Apply any snapshot that was staged on the previous rerun (before widgets instantiate)
if "_pending_load" in st.session_state:
    _staged = st.session_state.pop("_pending_load")
    for k, v in _staged.items():
        st.session_state[k] = v

SAVE_KEYS = [
    "num_variations",
    "users_c","conv_c","rev_c","prod_c",
    "users_v0","conv_v0","rev_v0","prod_v0",
    "users_v1","conv_v1","rev_v1","prod_v1",
    "users_v2","conv_v2","rev_v2","prod_v2",
    "days","conf_level","mc_method","primary_goal",
    # Sample-size calculator inputs (preserved in snapshots)
    "p_traffic","p_base_cr","p_base_aov","p_mde","p_vol",
    "s1_uc","s1_cc","s1_uv","s1_cv",
    "s2_uc","s2_cc","s2_uv","s2_cv",
    # Guardrail metrics
    "num_guardrails",
    "g0_name","g0_ctrl","g0_var","g0_threshold","g0_dir",
    "g1_name","g1_ctrl","g1_var","g1_threshold","g1_dir",
    "g2_name","g2_ctrl","g2_var","g2_threshold","g2_dir",
    # Segment breakdown
    "num_segments",
    "seg0_name","seg0_uc","seg0_cc","seg0_rc","seg0_uv","seg0_cv","seg0_rv",
    "seg1_name","seg1_uc","seg1_cc","seg1_rc","seg1_uv","seg1_cv","seg1_rv",
    "seg2_name","seg2_uc","seg2_cc","seg2_rc","seg2_uv","seg2_cv","seg2_rv",
    "seg3_name","seg3_uc","seg3_cc","seg3_rc","seg3_uv","seg3_cv","seg3_rv",
]


# -----------------------------------------------
# HELPERS
# -----------------------------------------------
def render_header(svg, text, level=2):
    st.markdown(
        f'<div style="display:flex;align-items:center;margin-bottom:10px;">'
        f'{svg}<h{level} style="margin:0;padding:0;">{text}</h{level}></div>',
        unsafe_allow_html=True,
    )

def safe_divide(n, d, fallback=0.0):
    return n / d if d != 0 else fallback

def calculate_uplift(ctrl, var):
    return safe_divide((var - ctrl) * 100, ctrl)


# -----------------------------------------------
# SRM TEST  (supports N groups)
# -----------------------------------------------
def perform_srm_test(observed, expected_split=None):
    n = len(observed)
    if expected_split is None:
        expected_split = [1 / n] * n
    total    = sum(observed)
    expected = [total * p for p in expected_split]
    stat, p  = chisquare(observed, f_exp=expected)
    return stat, p


# -----------------------------------------------
# DURATION ANALYSIS
# -----------------------------------------------
def analyze_test_duration(days, start_date=None):
    """
    Runs multiple business-cycle checks on the test duration.
    Returns a list of check dicts:
      {"id": str, "level": "pass"|"warning"|"error", "label": str, "msg": str}
    start_date: datetime.date or None. When provided, enables day-of-week bias detection.
    """
    _DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    full_weeks = days // 7
    remainder  = days % 7
    checks = []

    # 1. Minimum duration / novelty
    if days < 7:
        checks.append({"id": "too_short", "level": "error",
                        "label": "Too Short",
                        "msg": f"{days} day{'s' if days != 1 else ''} is below the 7-day minimum — results are unreliable."})
    elif days < 14:
        checks.append({"id": "novelty", "level": "warning",
                        "label": "Novelty Risk",
                        "msg": f"{days} days — users may still be reacting to the change. Industry standard is 14+ days."})
    else:
        checks.append({"id": "duration_ok", "level": "pass",
                        "label": "Duration OK",
                        "msg": f"{days} days ({full_weeks} full week{'s' if full_weeks != 1 else ''})."})

    # 2. Seasonality risk for very long tests
    if days > 90:
        checks.append({"id": "long_running", "level": "warning",
                        "label": "Seasonality Risk",
                        "msg": f"{days} days is a long test. Results may be contaminated by seasonal trends unrelated to the change."})

    # 3. Business week integrity
    if days >= 7 and remainder != 0:
        checks.append({"id": "incomplete_weeks", "level": "warning",
                        "label": "Incomplete Weeks",
                        "msg": (f"{days} days = {full_weeks} full week{'s' if full_weeks != 1 else ''}"
                                f" + {remainder} extra day{'s' if remainder != 1 else ''}."
                                f" Partial weeks introduce day-of-week bias.")})
    elif days >= 14:
        checks.append({"id": "weeks_ok", "level": "pass",
                        "label": "Full Weeks",
                        "msg": f"{full_weeks} complete week{'s' if full_weeks != 1 else ''} — no day-of-week bias."})

    # 4. Day-of-week alignment (only when start date is provided)
    if start_date is not None:
        end_date  = start_date + datetime.timedelta(days=days - 1)
        start_dow = start_date.weekday()   # 0 = Monday
        end_dow   = end_date.weekday()     # 6 = Sunday
        if start_dow == 0 and end_dow == 6:
            checks.append({"id": "dow_ok", "level": "pass",
                            "label": "Week Alignment",
                            "msg": f"Started {_DOW[start_dow]}, ended {_DOW[end_dow]} — perfect Mon→Sun alignment."})
        else:
            checks.append({"id": "dow_bias", "level": "warning",
                            "label": "Day-of-Week Bias",
                            "msg": (f"Test ran {_DOW[start_dow]} → {_DOW[end_dow]}. "
                                    f"Ideal window is Mon → Sun. Partial week coverage may skew results.")})

    return checks


# -----------------------------------------------
# GUARDRAIL METRICS ENGINE
# -----------------------------------------------
def evaluate_guardrails(guardrails):
    """
    guardrails: list of dicts — name, ctrl_val, var_val, threshold, direction.
    direction: "lower is better" | "higher is better"
    Returns list of result dicts with violated flag.
    """
    results = []
    for g in guardrails:
        ctrl = g["ctrl_val"]
        var  = g["var_val"]
        if ctrl == 0:
            results.append({
                "name": g["name"], "ctrl": ctrl, "var": var,
                "delta_pct": 0.0, "violated": False, "skip": True,
            })
            continue
        delta_pct = ((var - ctrl) / ctrl) * 100
        if g["direction"] == "lower is better":
            violated = delta_pct > g["threshold"]
        else:
            violated = delta_pct < -g["threshold"]
        results.append({
            "name":      g["name"],
            "ctrl":      ctrl,
            "var":       var,
            "delta_pct": delta_pct,
            "threshold": g["threshold"],
            "direction": g["direction"],
            "violated":  violated,
            "skip":      False,
        })
    return results


# -----------------------------------------------
# SEGMENT BREAKDOWN ENGINE
# -----------------------------------------------
def analyze_segments(segments, alpha):
    """
    segments: list of dicts — name, uc, cc, rc, uv, cv, rv.
    Returns one result dict per segment. No MCC — exploratory consistency check.
    """
    results = []
    for s in segments:
        uc, cc, rc = s["uc"], s["cc"], s["rc"]
        uv, cv, rv = s["uv"], s["cv"], s["rv"]
        if uc == 0 or uv == 0:
            results.append({"name": s["name"], "skip": True})
            continue
        ctrl_cr  = min(safe_divide(cc, uc), 1.0)
        var_cr   = min(safe_divide(cv, uv), 1.0)
        ctrl_rpv = safe_divide(rc, uc)
        var_rpv  = safe_divide(rv, uv)
        try:
            from statsmodels.stats.proportion import proportions_ztest
            z_stat, p_val = proportions_ztest([cc, cv], [uc, uv])
        except Exception:
            z_stat, p_val = 0.0, 1.0
        if not np.isfinite(p_val):
            p_val = 1.0
        results.append({
            "name":        s["name"],
            "ctrl_cr":     ctrl_cr,
            "var_cr":      var_cr,
            "ctrl_rpv":    ctrl_rpv,
            "var_rpv":     var_rpv,
            "uplift_cr":   calculate_uplift(ctrl_cr, var_cr),
            "uplift_rpv":  calculate_uplift(ctrl_rpv, var_rpv),
            "z_stat":      z_stat if np.isfinite(z_stat) else 0.0,
            "p_value":     p_val,
            "significant": p_val <= alpha,
            "ctrl_users":  uc,
            "var_users":   uv,
            "ctrl_conv":   cc,
            "var_conv":    cv,
            "skip":        False,
        })
    return results


# -----------------------------------------------
# MULTI-VARIANT STATISTICAL ENGINE
# -----------------------------------------------
def run_multivariate_analysis(groups, alpha, mc_method, primary_goal="Maximize CR"):
    """
    groups: list of dicts — name, users, conv, rev, prod.
    Returns omnibus chi-square, pairwise comparisons with MCC, per-group
    metrics, and the name of the winner (if any).

    Multiple comparison correction options:
      holm       – Holm-Bonferroni  (controls FWER, recommended default)
      bonferroni – Bonferroni        (most conservative)
      fdr_bh     – Benjamini-Hochberg (controls FDR, best for exploratory)
    """
    ctrl = groups[0]
    variations = groups[1:]

    # --- Per-group metrics ---
    ctrl_cr  = safe_divide(ctrl["conv"],  ctrl["users"])
    ctrl_aov = safe_divide(ctrl["rev"],   ctrl["conv"])
    ctrl_rpv = safe_divide(ctrl["rev"],   ctrl["users"])

    metrics = []
    for g in groups:
        cr  = min(safe_divide(g["conv"],  g["users"]), 1.0)   # clamp: conv can't exceed users
        aov = safe_divide(g["rev"],   g["conv"])
        rpv = safe_divide(g["rev"],   g["users"])
        apo = safe_divide(g["prod"],  g["conv"])
        apu = safe_divide(g["prod"],  g["users"])
        metrics.append({
            "name": g["name"], "users": g["users"], "conv": g["conv"],
            "cr": cr, "cr_pct": cr * 100, "aov": aov, "rpv": rpv, "apo": apo, "apu": apu,
            "uplift_cr":  calculate_uplift(ctrl_cr,  cr),
            "uplift_aov": calculate_uplift(ctrl_aov, aov),
            "uplift_rpv": calculate_uplift(ctrl_rpv, rpv),
        })

    # --- Omnibus chi-square (all groups simultaneously) ---
    conv_arr     = [g["conv"]                        for g in groups]
    non_conv_arr = [g["users"] - g["conv"]           for g in groups]
    try:
        chi2_g, p_global, _, _ = chi2_contingency(
            np.array([conv_arr, non_conv_arr]).T   # shape (n_groups, 2)
        )
    except Exception:
        chi2_g, p_global = 0.0, 1.0

    # --- Pairwise: each variation vs control ---
    raw_p, z_stats, labels = [], [], []
    for var in variations:
        try:
            z, p = proportions_ztest(
                [ctrl["conv"], var["conv"]],
                [ctrl["users"], var["users"]],
            )
        except Exception:
            z, p = 0.0, 1.0
        raw_p.append(p)
        z_stats.append(z)
        labels.append(var["name"])

    # --- Multiple comparison correction ---
    # Replace NaN/inf p-values (from degenerate inputs like conv > users) with 1.0
    safe_p = [p if np.isfinite(p) else 1.0 for p in raw_p]
    if len(safe_p) > 0:
        reject, p_adj, _, _ = multipletests(safe_p, alpha=alpha, method=mc_method)
    else:
        reject, p_adj = np.array([]), np.array([])

    ctrl_apo = metrics[0]["apo"]
    pairwise = []
    for i, var in enumerate(variations):
        m = metrics[i + 1]
        pairwise.append({
            "name":        var["name"],
            "z_stat":      z_stats[i],
            "p_raw":       raw_p[i],
            "p_adjusted":  float(p_adj[i]),
            "significant": bool(reject[i]),
            "uplift_cr":   m["uplift_cr"],
            "uplift_aov":  m["uplift_aov"],
            "uplift_rpv":  m["uplift_rpv"],
            "uplift_apo":  calculate_uplift(ctrl_apo, m["apo"]),
        })

    # --- Winner selection — behaviour driven by primary_goal ---
    #
    # "Maximize CR":      Only variants with CR uplift > 0 qualify.
    #                     Tiebreaker: RPV → AOV → lowest APO.
    #
    # "Maximize Revenue": Only variants with RPV uplift > 0 qualify.
    #                     A lower-CR variant can win if it earns more per visitor.
    #                     Tiebreaker: CR → AOV → lowest APO.
    #
    # "Balanced":         Composite score = 0.4 × uplift_cr + 0.6 × uplift_rpv.
    #                     Variant must be significant and have positive composite score.
    #                     Tiebreaker: AOV → lowest APO.

    _composite = lambda x: 0.4 * x["uplift_cr"] + 0.6 * x["uplift_rpv"]

    if primary_goal == "Maximize Revenue":
        sig_winners   = [pw for pw in pairwise if pw["significant"] and pw["uplift_rpv"] > 0]
        clean_winners = [pw for pw in sig_winners if pw["uplift_cr"] >= 0]
        _winner_key   = lambda x: (x["uplift_rpv"], x["uplift_cr"], x["uplift_aov"], -x["uplift_apo"])
    elif primary_goal == "Balanced":
        sig_winners   = [pw for pw in pairwise if pw["significant"] and _composite(pw) > 0]
        clean_winners = [pw for pw in sig_winners if pw["uplift_cr"] >= 0 and pw["uplift_rpv"] >= 0]
        _winner_key   = lambda x: (_composite(x), x["uplift_aov"], -x["uplift_apo"])
    else:  # "Maximize CR" (default)
        sig_winners   = [pw for pw in pairwise if pw["significant"] and pw["uplift_cr"] > 0]
        clean_winners = [pw for pw in sig_winners if pw["uplift_rpv"] >= 0]
        _winner_key   = lambda x: (x["uplift_cr"], x["uplift_rpv"], x["uplift_aov"], -x["uplift_apo"])

    winner = (
        max(clean_winners, key=_winner_key)["name"]
        if clean_winners
        else (max(sig_winners, key=_winner_key)["name"] if sig_winners else None)
    )
    # Flag if winner was chosen despite the "other" metric being negative
    winner_rpv_negative = (
        winner is not None and not clean_winners and bool(sig_winners)
    )

    return {
        "chi2_global": chi2_g, "p_global": p_global,
        "pairwise": pairwise, "metrics": metrics,
        "winner": winner, "winner_rpv_negative": winner_rpv_negative,
        "correction": mc_method,
        "n_comparisons": len(variations),
    }


# -----------------------------------------------
# BAYESIAN  (supports N groups)
# -----------------------------------------------
def calculate_bayesian_multivariate(groups):
    """
    Monte Carlo Beta-posterior analysis for all groups simultaneously.
    Returns prob_best and expected_loss per group.
    """
    rng = np.random.default_rng()
    samples = {}
    for g in groups:
        a = g["conv"] + 1
        b = max(g["users"] - g["conv"], 0) + 1
        samples[g["name"]] = rng.beta(a, b, BAYESIAN_SAMPLES)

    names = list(samples.keys())
    arr   = np.stack([samples[n] for n in names], axis=1)   # (SAMPLES, n_groups)

    best_idx   = np.argmax(arr, axis=1)
    prob_best  = {n: float(np.mean(best_idx == i)) for i, n in enumerate(names)}
    max_all    = arr.max(axis=1, keepdims=True)
    exp_loss   = {
        n: float(np.mean(np.maximum(max_all[:, 0] - arr[:, i], 0)))
        for i, n in enumerate(names)
    }
    return {"prob_best": prob_best, "expected_loss": exp_loss}

# Keep 2-group version for bootstrap tab
def calculate_bayesian_risk(ac, bc, av, bv):
    rng   = np.random.default_rng()
    s_c   = rng.beta(ac, bc, BAYESIAN_SAMPLES)
    s_v   = rng.beta(av, bv, BAYESIAN_SAMPLES)
    p_win = float(np.mean(s_v > s_c))
    lv    = float(np.mean(np.maximum(s_c - s_v, 0)))
    lc    = float(np.mean(np.maximum(s_v - s_c, 0)))
    return p_win, lv, lc


# -----------------------------------------------
# REVENUE SIGNIFICANCE
# -----------------------------------------------
def reconstruct_order_values(n_users, n_conv, total_rev, rng):
    if n_conv <= 0 or total_rev <= 0:
        return np.zeros(max(n_users, 1))
    n_users = max(n_users, 1)
    n_conv  = min(n_conv, n_users)   # conv can never exceed users — clamp defensively
    aov   = total_rev / n_conv
    sigma = 0.8
    mu    = np.log(aov) - (sigma ** 2) / 2
    vals  = rng.lognormal(mean=mu, sigma=sigma, size=n_conv)
    vals  = vals * (total_rev / vals.sum())
    arr   = np.zeros(n_users)
    idx   = rng.choice(n_users, size=n_conv, replace=False)
    arr[idx] = vals
    return arr

def test_revenue_significance(uc, cc, rc, uv, cv, rv, alpha=0.05, n_boot=2000):
    rng = np.random.default_rng(seed=42)
    aov_c = reconstruct_order_values(cc, cc, rc, rng) if cc > 0 else np.zeros(1)
    aov_v = reconstruct_order_values(cv, cv, rv, rng) if cv > 0 else np.zeros(1)
    rpv_c = reconstruct_order_values(uc, cc, rc, rng)
    rpv_v = reconstruct_order_values(uv, cv, rv, rng)
    results = {}
    for label, ac, av in [("aov", aov_c, aov_v), ("rpv", rpv_c, rpv_v)]:
        try:
            _, mw_p = mannwhitneyu(ac, av, alternative="two-sided") if (ac.sum() + av.sum()) > 0 else (0, 1.0)
        except ValueError:
            mw_p = 1.0
        lc, lv    = np.log1p(ac), np.log1p(av)
        diffs     = np.array([
            np.expm1(rng.choice(lv, len(lv), replace=True).mean()) -
            np.expm1(rng.choice(lc, len(lc), replace=True).mean())
            for _ in range(n_boot)
        ])
        ci_l  = float(np.percentile(diffs, alpha / 2 * 100))
        ci_h  = float(np.percentile(diffs, (1 - alpha / 2) * 100))
        od    = av.mean() - ac.mean()
        bp    = min(float(np.mean(diffs <= 0) * 2) if od >= 0 else float(np.mean(diffs >= 0) * 2), 1.0)
        results[label] = {
            "mw_p": mw_p, "mw_sig": mw_p <= alpha,
            "boot_p": bp, "boot_sig": bp <= alpha,
            "boot_ci_low": ci_l, "boot_ci_high": ci_h,
            "sig": (mw_p <= alpha) or (bp <= alpha),
        }
    return results


# -----------------------------------------------
# SIMPSON'S PARADOX
# -----------------------------------------------
def check_simpsons_paradox(seg1, seg2):
    cr_c1  = safe_divide(seg1["conv_c"], seg1["users_c"])
    cr_v1  = safe_divide(seg1["conv_v"], seg1["users_v"])
    up1    = calculate_uplift(cr_c1, cr_v1)
    cr_c2  = safe_divide(seg2["conv_c"], seg2["users_c"])
    cr_v2  = safe_divide(seg2["conv_v"], seg2["users_v"])
    up2    = calculate_uplift(cr_c2, cr_v2)
    agg_cc = seg1["conv_c"]  + seg2["conv_c"]
    agg_uc = seg1["users_c"] + seg2["users_c"]
    agg_cv = seg1["conv_v"]  + seg2["conv_v"]
    agg_uv = seg1["users_v"] + seg2["users_v"]
    up_agg = calculate_uplift(safe_divide(agg_cc, agg_uc), safe_divide(agg_cv, agg_uv))
    paradox = (up1 > 0 and up2 > 0 and up_agg < 0) or (up1 < 0 and up2 < 0 and up_agg > 0)
    return paradox, up1, up2, up_agg


# -----------------------------------------------
# AI ANALYSIS
# -----------------------------------------------
def get_ai_analysis(api_key, hypothesis, metrics, provider="OpenAI", conf_level="95%", segment_results=None):
    if not api_key:
        return "Please enter a valid API Key to generate this analysis."
    # Use None for OpenAI so the SDK uses its own default endpoint (future-proof)
    base_url   = "https://api.deepseek.com" if provider == "DeepSeek" else None
    model_name = "deepseek-reasoner"         if provider == "DeepSeek" else "gpt-4o"
    prompt = f"""You are an expert CRO analyst writing a structured A/B/n test report for a business stakeholder.

EXPERIMENT CONFIGURATION
- Confidence Level: {conf_level}
- Hypothesis: "{hypothesis}"

EXPERIMENT DATA
{json.dumps(metrics, indent=2, cls=NumpyEncoder)}

INSTRUCTIONS
Write in Markdown using exactly these sections. Do not restate the hypothesis as a heading — weave it into the Executive Summary opening sentence. If multiple variants are present, compare each vs control and identify a clear winner or explain why none qualifies.

## Executive Summary
One paragraph. State what was tested, whether any variant won, the best CR uplift, significance, and financial impact per visitor.

## Trade-off Analysis
CR vs AOV vs RPV across all variants. Identify clean wins, volume-vs-value trade-offs, or net negatives. Note whether MCC affected any conclusions.

## Risk Assessment
Bayesian probability and expected loss per variant. Factor in duration, SRM status, and number of variants tested.

## Visual Insights

### Strategic Matrix
Where each variant sits on CR vs AOV quadrant and what it means strategically.

### Product Metrics
Basket size trends (APO, APU) across all groups and behavioural implications.

### Revenue Charts
RPV and AOV side by side — what the combination tells us about revenue quality per variant.

### CR Comparison
Contextualise each variant's uplift given sample size and base rate.

### Bayesian Posterior
Overlapping Beta distributions — overlap degree, certainty, and probability each variant is best.

### Bootstrap Distribution
Histogram of resampled CR differences — centre, CI bounds, zero-crossing implications.

### Box Plot
Median, IQR spread, and stability across groups.

## Recommendation
One paragraph. Ship / do not ship / run longer for each variant. Name a winner if one exists. End with one concrete next step."""

    active_segs = [s for s in (segment_results or []) if not s.get("skip")]
    if active_segs:
        seg_lines = "\n".join(
            f"  - {s['name']}: Ctrl CR {s['ctrl_cr']*100:.2f}% → Var CR {s['var_cr']*100:.2f}% "
            f"(CR {s['uplift_cr']:+.2f}%, RPV {s['uplift_rpv']:+.2f}%, p={s['p_value']:.4f})"
            for s in active_segs
        )
        prompt += f"""

## Segment Breakdown
For each segment below, comment on whether the result holds, whether there are meaningful differences across audiences, and any risk of the aggregate result masking a losing segment.

Segment data (Control vs Best Variation):
{seg_lines}"""

    try:
        client   = openai.OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"**Error connecting to AI provider:** {e}"


# -----------------------------------------------
# SMART ANALYSIS  (rule-based)
# -----------------------------------------------
def generate_smart_analysis(hypothesis, mv_results, bayes_mv, metrics_payload, alpha_val, segment_results=None):
    report = []
    conf_pct = f"{(1 - alpha_val) * 100:.0f}%"

    # ── CRITICAL: SRM blocks all further analysis ──────────────────────────
    if metrics_payload["p_srm"] < 0.01:
        report.append("### CRITICAL — Sample Ratio Mismatch")
        report.append(
            f"SRM p = **{metrics_payload['p_srm']:.4f}**. Traffic split is uneven — "
            "**all results below are unreliable.** Fix randomisation before drawing any conclusions."
        )
        return "\n\n".join(report)

    n_vars       = mv_results["n_comparisons"]
    winner       = mv_results["winner"]
    p_glob       = mv_results["p_global"]
    primary_goal = metrics_payload.get("primary_goal", "Maximize CR")
    rev_sig      = metrics_payload.get("rev_sig", {})
    guardrails   = metrics_payload.get("guardrail_results", [])
    all_metrics  = metrics_payload.get("metrics", [])

    # ── HEADLINE ────────────────────────────────────────────────────────────
    if n_vars == 1:
        pw = mv_results["pairwise"][0]
        p_display = f"{pw['p_adjusted']:.4f}" if np.isfinite(pw['p_adjusted']) else "—"
        if pw["significant"] and pw["uplift_cr"] > 0:
            headline = "WINNER — Statistically Significant Positive Result"
            summary  = f"Variation outperforms Control at {conf_pct} confidence (p_adj = {p_display})."
        elif pw["significant"] and pw["uplift_cr"] < 0:
            headline = "LOSER — Statistically Significant Negative Result"
            summary  = f"Variation is significantly worse than Control (p_adj = {p_display}). Do not ship."
        else:
            headline = "INCONCLUSIVE — No Clear Winner"
            summary  = f"Cannot reject the null hypothesis (p_adj = {p_display}). More data needed."
    else:
        if winner:
            headline = f"MULTI-VARIANT WINNER — {winner}"
            summary  = (
                f"Omnibus test confirms a significant difference (χ² p = {p_glob:.4f}). "
                f"**{winner}** is the strongest performer after {mv_results['correction'].upper()} correction."
            )
        elif p_glob <= alpha_val:
            headline = "SIGNIFICANT DIFFERENCE — No Positive Winner"
            summary  = (
                f"Omnibus test significant (p = {p_glob:.4f}) but no variant shows a "
                "significant positive uplift after correction."
            )
        else:
            headline = "MULTI-VARIANT INCONCLUSIVE"
            summary  = f"No significant overall difference (omnibus p = {p_glob:.4f})."

    report.append(f"### {headline}")
    report.append(summary)
    if hypothesis:
        report.append(f"**Hypothesis:** _{hypothesis}_")
    report.append(f"_Optimisation goal: **{primary_goal}** · Confidence level: **{conf_pct}**_")

    # ── DATA HEALTH ─────────────────────────────────────────────────────────
    report.append("### Data Health & Validity")
    days = metrics_payload["days"]
    dur_status = "WARNING:" if days < 14 else "PASS:"
    dur_note   = f"Only {days} days — too short for reliable results." if days < 7 else \
                 f"{days} days — watch for novelty effects." if days < 14 else \
                 f"{days} days — healthy duration."
    report.append(f"- **{dur_status}** Duration: {dur_note}")
    report.append(f"- **PASS:** SRM test (p = {metrics_payload['p_srm']:.4f}) — traffic split is even.")
    if n_vars > 1:
        report.append(
            f"- **INFO:** {mv_results['correction'].upper()} multiple comparison correction "
            f"applied across {n_vars} pairwise comparisons."
        )

    # ── PAIRWISE RESULTS ────────────────────────────────────────────────────
    report.append("### Pairwise Results vs Control")
    for pw in mv_results["pairwise"]:
        sig   = "Significant" if pw["significant"] else "Not Significant"
        cr_d  = "▲" if pw["uplift_cr"]  >= 0 else "▼"
        rpv_d = "▲" if pw["uplift_rpv"] >= 0 else "▼"
        aov_d = "▲" if pw["uplift_aov"] >= 0 else "▼"
        p_display = f"{pw['p_adjusted']:.4f}" if np.isfinite(pw['p_adjusted']) else "—"
        report.append(
            f"- **{pw['name']}** [{sig}] — "
            f"CR {cr_d}{pw['uplift_cr']:+.2f}% | "
            f"RPV {rpv_d}{pw['uplift_rpv']:+.2f}% | "
            f"AOV {aov_d}{pw['uplift_aov']:+.2f}% | "
            f"p_adj = {p_display}"
        )

    # ── PRODUCT VELOCITY ────────────────────────────────────────────────────
    if len(all_metrics) > 1:
        report.append("### Product Velocity")
        for m in all_metrics[1:]:
            apo_d = "▲" if m.get("uplift_apo", 0) >= 0 else "▼"
            note  = ""
            if abs(m.get("uplift_apo", 0)) > 10:
                if m["uplift_apo"] < 0 and m["uplift_rpv"] >= 0:
                    note = " — fewer items, same or higher revenue: better per-item margin."
                elif m["uplift_apo"] > 0 and m["uplift_rpv"] >= 0:
                    note = " — more items per order with higher revenue: basket expansion."
                elif m["uplift_apo"] > 0 and m["uplift_rpv"] < 0:
                    note = " — more items per order but lower revenue: possible discounting risk."
            apo_val = m.get("uplift_apo", 0)
            report.append(
                f"- **{m['name']}**: {m['apo']:.2f} items/order "
                f"({apo_d}{apo_val:+.1f}% vs Control){note}"
            )

    # ── REVENUE SIGNALS ─────────────────────────────────────────────────────
    if rev_sig:
        report.append("### Revenue Signals")
        report.append(
            "_Based on reconstructed order distributions — treat as directional signals, not precise p-values._"
        )
        for label, display in [("rpv", "Revenue Per Visitor"), ("aov", "Average Order Value")]:
            r = rev_sig.get(label, {})
            if not r:
                continue
            overall = "Significant" if r.get("sig") else "Not significant"
            mw_p    = f"{r['mw_p']:.4f}" if np.isfinite(r.get("mw_p", float("nan"))) else "—"
            ci_l    = r.get("boot_ci_low",  0)
            ci_h    = r.get("boot_ci_high", 0)
            ci_note = "CI entirely positive — gain is consistent." if ci_l > 0 else \
                      "CI entirely negative — loss is consistent." if ci_h < 0 else \
                      "CI crosses zero — result is uncertain."
            report.append(
                f"- **{display}:** {overall} (Mann-Whitney p = {mw_p}) · "
                f"Bootstrap CI: ${ci_l:.3f} to ${ci_h:.3f} · {ci_note}"
            )

    # ── BAYESIAN ASSESSMENT ─────────────────────────────────────────────────
    report.append("### Bayesian Assessment")
    best_bayes = max(bayes_mv["prob_best"], key=bayes_mv["prob_best"].get)
    for name, prob in bayes_mv["prob_best"].items():
        loss = bayes_mv["expected_loss"].get(name, 0)
        tag  = " ← highest probability" if name == best_bayes else ""
        interp = ""
        if prob >= 0.95:
            interp = "Strong Bayesian evidence."
        elif prob >= 0.80:
            interp = "Moderate evidence — worth monitoring."
        elif prob >= 0.60:
            interp = "Weak signal — inconclusive."
        else:
            interp = "No meaningful advantage."
        report.append(
            f"- **{name}**: {prob*100:.1f}% P(best) | "
            f"Expected loss if wrong: {loss*100:.4f}%{tag} — {interp}"
        )

    # ── GUARDRAIL STATUS ────────────────────────────────────────────────────
    active_guards = [g for g in guardrails if not g.get("skip")]
    if active_guards:
        report.append("### Guardrail Metrics")
        any_violated = any(g["violated"] for g in active_guards)
        if any_violated:
            report.append("**WARNING: One or more guardrail metrics are violated.**")
        else:
            report.append("All guardrail metrics are within acceptable thresholds.")
        for g in active_guards:
            status = "VIOLATED" if g["violated"] else "PASS"
            d = "▲" if g["delta_pct"] > 0 else "▼"
            report.append(
                f"- **{g['name']}** [{status}]: {d}{abs(g['delta_pct']):.1f}% change "
                f"(limit ±{g['threshold']}%, {g['direction']})"
            )

    # ── SEGMENT BREAKDOWN ────────────────────────────────────────────────────
    active_segs = [s for s in (segment_results or []) if not s.get("skip")]
    if active_segs:
        report.append("### Segment Breakdown")
        wins  = sum(1 for s in active_segs if s["uplift_cr"] > 0)
        total = len(active_segs)
        if wins > total / 2:
            report.append(f"Variation leads in **{wins}/{total}** segments — result is consistent.")
        elif wins == total / 2:
            report.append(f"Variation leads in **{wins}/{total}** segments — mixed result.")
        else:
            report.append(
                f"Variation leads in only **{wins}/{total}** segments — "
                "aggregate uplift may not generalise across all audiences."
            )
        for s in active_segs:
            sig_tag = " [Sig]" if s["significant"] else ""
            cr_d  = "▲" if s["uplift_cr"]  >= 0 else "▼"
            rpv_d = "▲" if s["uplift_rpv"] >= 0 else "▼"
            report.append(
                f"- **{s['name']}**{sig_tag}: "
                f"CR {cr_d}{s['uplift_cr']:+.2f}% | "
                f"RPV {rpv_d}{s['uplift_rpv']:+.2f}% | "
                f"p = {s['p_value']:.4f}"
            )

    # ── STRATEGIC CONCLUSION ────────────────────────────────────────────────
    report.append("### Strategic Conclusion")
    if winner:
        pw_w         = next(p for p in mv_results["pairwise"] if p["name"] == winner)
        cr_up        = pw_w["uplift_cr"]
        rpv_up       = pw_w["uplift_rpv"]
        aov_up       = pw_w["uplift_aov"]
        guards_ok    = not any(g["violated"] for g in active_guards) if active_guards else True
        guard_note   = " Guardrail check: PASS." if guards_ok and active_guards else \
                       " **Guardrail check: FAILED — review secondary metrics before shipping.**" if active_guards else ""

        if cr_up > 0 and rpv_up > 0 and aov_up >= 0:
            verdict = f"**[SHIP] {winner} — Growth Engine.** CR, RPV, and AOV all positive."
        elif cr_up > 0 and rpv_up > 0 and aov_up < 0:
            verdict = f"**[SHIP WITH CAUTION] {winner} — Volume Play.** CR and RPV up, but AOV down — higher conversion from lower-value orders."
        elif cr_up <= 0 and rpv_up > 0:
            verdict = f"**[REVIEW] {winner} — Quality Play.** Lower CR but higher RPV — fewer, higher-value conversions. Consistent with '{primary_goal}' goal."
        elif cr_up > 0 and rpv_up < 0:
            verdict = f"**[REVIEW] {winner} — Volume Play.** CR up but RPV down — watch margin impact before shipping."
        else:
            verdict = f"**[REVIEW] {winner} — Mixed Signals.** Review all metrics before deciding."
        report.append(verdict + guard_note)
    else:
        report.append(
            "**[DO NOT SHIP]** No variant qualifies on current data. "
            "Consider running longer to accumulate statistical power, or revisit the hypothesis."
        )

    return "\n\n".join(report)


# -----------------------------------------------
# PDF REPORT GENERATOR
# -----------------------------------------------
def generate_pdf_report(mv, bayes, rev_sig, guardrail_results, duration_checks,
                         p_srm, groups, days_run, confidence_level, primary_goal,
                         ctrl_m, best_m, hypothesis="", smart_text="",
                         ai_text="", segment_results=None):
    """Build a PDF report from the current analysis state. Returns bytes."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                     TableStyle, HRFlowable, Image, PageBreak)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buf    = io.BytesIO()
    MARGIN = 18 * mm
    USABLE = A4[0] - 2 * MARGIN
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=MARGIN, rightMargin=MARGIN,
                                topMargin=MARGIN,  bottomMargin=MARGIN)

    # ── Palette ──────────────────────────────────────────────────
    C_RED   = colors.HexColor('#E73B37')
    C_DARK  = colors.HexColor('#1a1d24')
    C_LGRAY = colors.HexColor('#f5f5f5')
    C_MGRAY = colors.HexColor('#cccccc')
    C_GREEN = colors.HexColor('#2ca02c')
    C_WARN  = colors.HexColor('#d97706')
    C_BODY  = colors.HexColor('#333333')
    C_MUTED = colors.HexColor('#888888')

    # ── Style factory ─────────────────────────────────────────────
    def _s(name, size, color=C_DARK, bold=False, align=TA_LEFT, sb=0, sa=4):
        return ParagraphStyle(name, fontSize=size, textColor=color,
                               fontName='Helvetica-Bold' if bold else 'Helvetica',
                               alignment=align, spaceBefore=sb, spaceAfter=sa,
                               leading=size * 1.45)

    s_title = _s('pt',  20, C_DARK,  True,  TA_CENTER, 0, 8)
    s_h1    = _s('ph1', 13, C_DARK,  True,  TA_LEFT,  12, 5)
    s_h2    = _s('ph2', 10, C_RED,   True,  TA_LEFT,   6, 3)
    s_body  = _s('pb',   9, C_BODY,  False, TA_LEFT,   0, 2)
    s_small = _s('ps',   7, C_MUTED, False, TA_LEFT,   0, 1)
    s_win   = _s('pw',  10, colors.white, True, TA_CENTER, 0, 0)

    # ── Table helper ──────────────────────────────────────────────
    def _tbl(data, col_widths, extra=None):
        t  = Table(data, colWidths=col_widths)
        ts = TableStyle([
            ('BACKGROUND',    (0, 0), (-1,  0), C_DARK),
            ('TEXTCOLOR',     (0, 0), (-1,  0), colors.white),
            ('FONTNAME',      (0, 0), (-1,  0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0, 0), (-1, -1), 9),
            ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, C_LGRAY]),
            ('GRID',          (0, 0), (-1, -1), 0.5, C_MGRAY),
            ('PADDING',       (0, 0), (-1, -1), 5),
            ('ALIGN',         (1, 0), (-1, -1), 'CENTER'),
            ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ])
        for cmd in (extra or []):
            ts.add(*cmd)
        t.setStyle(ts)
        return t

    # ── Markdown → reportlab XML (handles **bold**) ───────────────
    def _md(text):
        text = text.replace('&', '&amp;')
        text = _re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        return text

    # ── Figure → in-memory PNG bytes ─────────────────────────────
    def _fig_bytes(fig):
        b = io.BytesIO()
        fig.savefig(b, format='png', bbox_inches='tight', dpi=110)
        b.seek(0)
        plt.close(fig)
        return b

    # Light rcParams for PDF charts (overrides global dark theme)
    _PDF_RC = {
        'figure.facecolor': 'white',   'axes.facecolor':  '#f8f8f8',
        'savefig.facecolor': 'white',  'text.color':      '#1a1d24',
        'axes.labelcolor':  '#1a1d24', 'axes.titlecolor': '#1a1d24',
        'xtick.color':      '#1a1d24', 'ytick.color':     '#1a1d24',
        'grid.color':       '#dddddd', 'axes.edgecolor':  '#cccccc',
        'legend.facecolor': 'white',   'legend.edgecolor':'#cccccc',
        'legend.labelcolor':'#1a1d24',
    }

    IMG_W   = USABLE
    IMG_H   = 82 * mm
    metrics = mv["metrics"]
    elems   = []

    # ── PAGE 1: HEADER & KEY METRICS ─────────────────────────────
    elems.append(Paragraph("A/B Test Analysis Report", s_title))
    elems.append(HRFlowable(width=USABLE, color=C_RED, thickness=2, spaceAfter=6))
    if hypothesis:
        elems.append(Paragraph(f"<i>Hypothesis: {_md(hypothesis)}</i>", s_body))
        elems.append(Spacer(1, 4))

    start_date = st.session_state.get("start_date")
    meta = [
        ["Generated",     datetime.date.today().isoformat()],
        ["Test Duration", f"{days_run} days" + (f"  (started {start_date})" if start_date else "")],
        ["Confidence",    confidence_level],
        ["Primary Goal",  primary_goal],
        ["Groups",        ", ".join(g["name"] for g in groups)],
    ]
    elems.append(_tbl(meta, [45 * mm, USABLE - 45 * mm]))
    elems.append(Spacer(1, 10))

    winner = mv["winner"]
    if winner:
        pw     = next((p for p in mv["pairwise"] if p["name"] == winner), {})
        banner = (f"WINNER: {winner}   |   CR {pw.get('uplift_cr', 0):+.2f}%"
                  f"   |   RPV {pw.get('uplift_rpv', 0):+.2f}%"
                  f"   |   p = {pw.get('p_adjusted', 1):.4f}")
        bg = C_GREEN
    else:
        banner = "No winner declared — test inconclusive or insufficient statistical power."
        bg     = C_DARK
    w_tbl = Table([[Paragraph(banner, s_win)]], colWidths=[USABLE])
    w_tbl.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,-1), bg),
                                ('PADDING',    (0,0), (-1,-1), 10)]))
    elems.append(w_tbl)
    elems.append(Spacer(1, 12))

    elems.append(Paragraph("Key Metrics", s_h1))
    n   = len(metrics)
    cw  = USABLE / (n + 1)
    hdr = ["Metric"] + [m["name"] for m in metrics]
    rows = [
        ["Users"]       + [f"{m['users']:,}" for m in metrics],
        ["Conversions"] + [f"{m['conv']:,}"  for m in metrics],
        ["CR %"]        + [f"{m['cr_pct']:.2f}%" + (f" ({m['uplift_cr']:+.1f}%)" if i > 0 else "")
                           for i, m in enumerate(metrics)],
        ["AOV"]         + [f"${m['aov']:.2f}" + (f" ({m['uplift_aov']:+.1f}%)" if i > 0 else "")
                           for i, m in enumerate(metrics)],
        ["RPV"]         + [f"${m['rpv']:.4f}" + (f" ({m['uplift_rpv']:+.1f}%)" if i > 0 else "")
                           for i, m in enumerate(metrics)],
    ]
    elems.append(_tbl([hdr] + rows, [cw] * (n + 1)))
    elems.append(Spacer(1, 12))

    # ── STATISTICAL ANALYSIS ─────────────────────────────────────
    elems.append(Paragraph("Statistical Analysis", s_h1))
    pw_hdr  = ["Variation", "Z-Stat", "p (adj)", "Significant", "CR Uplift", "RPV Uplift"]
    pw_rows = [[pw["name"], f"{pw['z_stat']:.3f}", f"{pw['p_adjusted']:.4f}",
                "YES" if pw["significant"] else "No",
                f"{pw['uplift_cr']:+.2f}%", f"{pw['uplift_rpv']:+.2f}%"]
               for pw in mv["pairwise"]]
    sig_ex  = [('TEXTCOLOR', (3, i+1), (3, i+1), C_GREEN if pw["significant"] else C_RED)
               for i, pw in enumerate(mv["pairwise"])]
    elems.append(_tbl([pw_hdr] + pw_rows, [USABLE / 6] * 6, sig_ex))
    elems.append(Spacer(1, 8))

    elems.append(Paragraph("Bayesian Results", s_h2))
    b_hdr  = ["Group", "P(Best)", "Expected Loss"]
    b_rows = [[g, f"{bayes['prob_best'][g] * 100:.1f}%",
               f"{bayes['expected_loss'][g] * 100:.4f}%"]
              for g in bayes["prob_best"]]
    elems.append(_tbl([b_hdr] + b_rows, [USABLE / 3] * 3))
    elems.append(Spacer(1, 8))

    elems.append(Paragraph("Revenue Significance (Control vs Best Variation)", s_h2))
    r_hdr  = ["Metric", "MW p-value", "Bootstrap p", "CI Low", "CI High", "Significant"]
    r_rows = [
        ["AOV", f"{rev_sig['aov']['mw_p']:.4f}", f"{rev_sig['aov']['boot_p']:.4f}",
         f"${rev_sig['aov']['boot_ci_low']:.2f}", f"${rev_sig['aov']['boot_ci_high']:.2f}",
         "YES" if rev_sig['aov']['sig'] else "No"],
        ["RPV", f"{rev_sig['rpv']['mw_p']:.4f}", f"{rev_sig['rpv']['boot_p']:.4f}",
         f"${rev_sig['rpv']['boot_ci_low']:.4f}", f"${rev_sig['rpv']['boot_ci_high']:.4f}",
         "YES" if rev_sig['rpv']['sig'] else "No"],
    ]
    r_ex = [('TEXTCOLOR', (5, i+1), (5, i+1), C_GREEN if sig else C_DARK)
            for i, sig in enumerate([rev_sig["aov"]["sig"], rev_sig["rpv"]["sig"]])]
    elems.append(_tbl([r_hdr] + r_rows, [USABLE / 6] * 6, r_ex))
    elems.append(PageBreak())

    # ── PAGE 2: HEALTH CHECKS ────────────────────────────────────
    elems.append(Paragraph("Health Checks", s_h1))

    elems.append(Paragraph("Sample Ratio Mismatch", s_h2))
    srm_ok  = p_srm >= 0.01
    srm_msg = (f"PASSED (p = {p_srm:.4f}) — Traffic split is even."
               if srm_ok else
               f"DETECTED (p = {p_srm:.4f}) — Traffic split is uneven. Results may be invalid.")
    elems.append(Paragraph(srm_msg, _s('srm', 9, C_GREEN if srm_ok else C_RED, True)))
    elems.append(Spacer(1, 6))

    elems.append(Paragraph("Duration Analysis", s_h2))
    _LVL = {"pass": C_GREEN, "warning": C_WARN, "error": C_RED}
    for chk in duration_checks:
        col = _LVL.get(chk["level"], C_DARK)
        elems.append(Paragraph(f"[{chk['label']}] {chk['msg']}",
                                _s(f'dc_{chk["id"]}', 9, col, chk["level"] != "pass")))
    elems.append(Spacer(1, 6))

    active_g = [r for r in guardrail_results if not r.get("skip")]
    if active_g:
        elems.append(Paragraph("Guardrail Metrics", s_h2))
        g_hdr  = ["Metric", "Control", "Variation", "Change", "Threshold", "Status"]
        g_rows = [[r["name"], f"{r['ctrl']:.2f}", f"{r['var']:.2f}",
                   f"{r['delta_pct']:+.2f}%", f"±{r['threshold']:.1f}%",
                   "VIOLATED" if r["violated"] else "PASS"]
                  for r in active_g]
        g_ex   = [('TEXTCOLOR', (5, i+1), (5, i+1), C_RED if r["violated"] else C_GREEN)
                  for i, r in enumerate(active_g)]
        elems.append(_tbl([g_hdr] + g_rows, [USABLE / 6] * 6, g_ex))

    active_pdf_segs = [s for s in (segment_results or []) if not s.get("skip")]
    if active_pdf_segs:
        elems.append(Spacer(1, 6))
        elems.append(Paragraph("Segment Breakdown", s_h2))
        sg_hdr  = ["Segment", "Ctrl CR", "Var CR", "CR Uplift", "Ctrl RPV", "Var RPV", "RPV Uplift", "p-value", "Sig"]
        sg_rows = [[
            s["name"],
            f"{s['ctrl_cr']*100:.2f}%", f"{s['var_cr']*100:.2f}%",
            f"{s['uplift_cr']:+.2f}%",
            f"${s['ctrl_rpv']:.4f}", f"${s['var_rpv']:.4f}",
            f"{s['uplift_rpv']:+.2f}%",
            f"{s['p_value']:.4f}",
            "YES" if s["significant"] else "No",
        ] for s in active_pdf_segs]
        sg_ex = [('TEXTCOLOR', (8, i+1), (8, i+1), C_GREEN if s["significant"] else C_DARK)
                 for i, s in enumerate(active_pdf_segs)]
        cw9 = USABLE / 9
        elems.append(_tbl([sg_hdr] + sg_rows, [cw9] * 9, sg_ex))

    elems.append(PageBreak())

    # ── PAGE 3: CHARTS ────────────────────────────────────────────
    elems.append(Paragraph("Charts", s_h1))
    with plt.rc_context(_PDF_RC):
        # CR Comparison
        cr_vals    = [m["cr_pct"] for m in metrics]
        cr_labels  = [m["name"]   for m in metrics]
        fig, ax    = plt.subplots(figsize=(10, 4))
        ctrl_val   = cr_vals[0]
        bar_colors = [GROUP_COLORS[i] if (i == 0 or v >= ctrl_val) else "#d62728"
                      for i, v in enumerate(cr_vals)]
        bars = ax.bar(cr_labels, cr_vals, color=bar_colors, alpha=0.85)
        mx   = max(cr_vals) if max(cr_vals) > 0 else 1
        ax.set_ylim(0, mx * 1.18)
        ax.set_title("Conversion Rate by Group")
        ax.set_ylabel("CR %")
        for bar, v in zip(bars, cr_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + mx * 0.01,
                    f"{v:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
        elems.append(Paragraph("Conversion Rate Comparison", s_h2))
        elems.append(Image(_fig_bytes(fig), width=IMG_W, height=IMG_H))
        elems.append(Spacer(1, 8))

        # Strategic Matrix
        fig, ax  = plt.subplots(figsize=(9, 5))
        ctrl_m0  = metrics[0]
        for i, m in enumerate(metrics):
            ax.scatter(m["cr_pct"], m["aov"], color=GROUP_COLORS[i], s=180,
                       label=m["name"], zorder=5)
            if i > 0:
                ax.annotate("", xy=(m["cr_pct"], m["aov"]),
                            xytext=(ctrl_m0["cr_pct"], ctrl_m0["aov"]),
                            arrowprops=dict(arrowstyle="->", color=GROUP_COLORS[i], lw=1.5, ls="--"))
        ax.axvline(ctrl_m0["cr_pct"], color="#888", ls=":", alpha=0.4)
        ax.axhline(ctrl_m0["aov"],    color="#888", ls=":", alpha=0.4)
        ax.set_title("Strategic Matrix: CR vs AOV")
        ax.set_xlabel("Conversion Rate (%)")
        ax.set_ylabel("Average Order Value ($)")
        ax.legend()
        elems.append(Paragraph("Strategic Matrix", s_h2))
        elems.append(Image(_fig_bytes(fig), width=IMG_W, height=IMG_H))
        elems.append(Spacer(1, 8))

        # Bayesian PDFs
        fig, ax = plt.subplots(figsize=(11, 4))
        for i, g in enumerate(groups):
            a_  = g["conv"] + 1
            b_  = max(g["users"] - g["conv"], 0) + 1
            d_  = beta(a_, b_)
            x_  = np.linspace(d_.ppf(0.001), d_.ppf(0.999), 1000)
            ax.plot(x_, d_.pdf(x_), label=g["name"], color=GROUP_COLORS[i], lw=2)
            ax.fill_between(x_, d_.pdf(x_), 0, alpha=0.15, color=GROUP_COLORS[i])
        ax.set_title("Bayesian Posterior Distributions")
        ax.set_xlabel("Conversion Rate")
        ax.set_ylabel("Probability Density")
        ax.legend()
        elems.append(Paragraph("Bayesian Posteriors", s_h2))
        elems.append(Image(_fig_bytes(fig), width=IMG_W, height=IMG_H))
        elems.append(Spacer(1, 8))

        # Bootstrap CI
        _rng  = np.random.default_rng(42)
        _pc   = float(np.clip(safe_divide(ctrl_m["conv"], ctrl_m["users"]), 0.0, 1.0))
        _pv   = float(np.clip(safe_divide(best_m["conv"], best_m["users"]), 0.0, 1.0))
        _sc   = _rng.binomial(ctrl_m["users"], _pc, BOOTSTRAP_SAMPLES) / ctrl_m["users"]
        _sv   = _rng.binomial(best_m["users"], _pv, BOOTSTRAP_SAMPLES) / best_m["users"]
        diffs = _sv - _sc
        ci_l  = float(np.percentile(diffs, 2.5))
        ci_h  = float(np.percentile(diffs, 97.5))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(diffs, bins=60, color="#ff7f0e", edgecolor="#cccccc", alpha=0.85)
        ax.axvline(ci_l, color="#E73B37", ls="--", lw=1.5, label="95% CI")
        ax.axvline(ci_h, color="#E73B37", ls="--", lw=1.5)
        ax.axvline(0,    color="#555555", lw=1.2, ls=":",  label="No effect")
        ax.set_title(f"Bootstrap CR Difference ({best_m['name']} − Control)")
        ax.set_xlabel("Difference in Conversion Rate")
        ax.set_ylabel("Frequency")
        ax.legend()
        elems.append(Paragraph("Bootstrap Confidence Interval", s_h2))
        elems.append(Image(_fig_bytes(fig), width=IMG_W, height=IMG_H))

        # Segment CR chart (if data present)
        if active_pdf_segs:
            seg_names  = [s["name"]        for s in active_pdf_segs]
            ctrl_cr_v  = [s["ctrl_cr"]*100 for s in active_pdf_segs]
            var_cr_v   = [s["var_cr"]*100  for s in active_pdf_segs]
            x = np.arange(len(seg_names))
            w = 0.35
            fig, ax = plt.subplots(figsize=(max(6, len(seg_names) * 1.8), 4))
            ax.bar(x - w/2, ctrl_cr_v, w, label="Control", color=GROUP_COLORS[0])
            var_col = [GROUP_COLORS[1] if v >= c else "#d62728"
                       for v, c in zip(var_cr_v, ctrl_cr_v)]
            for xi, (val, col) in enumerate(zip(var_cr_v, var_col)):
                ax.bar(xi + w/2, val, w, color=col)
            ax.bar([], [], color=GROUP_COLORS[1], label=best_m["name"])
            ax.set_xticks(x)
            ax.set_xticklabels(seg_names)
            ax.set_title("Segment Breakdown — Conversion Rate")
            ax.set_ylabel("CR (%)")
            ax.legend(fontsize=8)
            elems.append(Spacer(1, 8))
            elems.append(Paragraph("Segment Breakdown — CR", s_h2))
            elems.append(Image(_fig_bytes(fig), width=IMG_W, height=IMG_H))

    def _render_text_page(title, text):
        elems.append(PageBreak())
        elems.append(Paragraph(title, s_h1))
        elems.append(HRFlowable(width=USABLE, color=C_RED, thickness=1, spaceAfter=6))
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                elems.append(Spacer(1, 3))
            elif line.startswith("### "):
                elems.append(Paragraph(line[4:], s_h2))
            elif line.startswith("## "):
                elems.append(Paragraph(line[3:], _s('sh1x', 11, C_DARK, True, TA_LEFT, 8, 3)))
            elif line.startswith("# "):
                elems.append(Paragraph(line[2:], s_h1))
            elif line.startswith(("- ", "* ")):
                elems.append(Paragraph("• " + _md(line[2:]), s_body))
            else:
                try:
                    elems.append(Paragraph(_md(line), s_body))
                except Exception:
                    elems.append(Paragraph(line, s_body))

    # ── PAGE 4+ (optional): SMART ANALYSIS & AI ANALYSIS TEXT ───
    if smart_text:
        _render_text_page("Smart Analysis Report", smart_text)
    if ai_text:
        _render_text_page("AI Analysis Report", ai_text)

    # ── FOOTER ────────────────────────────────────────────────────
    elems.append(Spacer(1, 14))
    elems.append(HRFlowable(width=USABLE, color=C_MGRAY, thickness=0.5))
    elems.append(Paragraph(
        "Generated by Enterprise A/B Test Analyzer. "
        "Results are directional signals — validate data integrity before making business decisions.",
        s_small))

    doc.build(elems)
    buf.seek(0)
    return buf.getvalue()


# -----------------------------------------------
# PLOTTING
# -----------------------------------------------
def plot_multivariant_bar(title, values, labels, unit=""):
    ctrl_val = values[0]
    colors   = [
        GROUP_COLORS[i] if (i == 0 or v >= ctrl_val) else "#d62728"
        for i, v in enumerate(values)
    ]
    hover = [f"{unit}{v:.4f}" for v in values]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{unit}{v:.2f}" for v in values],
        textposition="outside",
        hovertext=hover,
        hovertemplate="%{x}: %{hovertext}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{title} by Group",
        yaxis_title=title,
        plot_bgcolor="#1a1d24",
        paper_bgcolor="#0e1117",
        font_color="#e0e0e0",
        title_font_size=15,
        yaxis=dict(gridcolor="#2e3140", showgrid=True),
        xaxis=dict(showgrid=False),
        margin=dict(t=50, b=40, l=50, r=20),
        hoverlabel=dict(bgcolor="#1a1d24", font_color="#e0e0e0"),
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_strategic_matrix(metrics_list):
    ctrl = metrics_list[0]
    cx, cy = ctrl["cr_pct"], ctrl["aov"]

    fig = go.Figure()

    # Quadrant annotation labels
    for label, ax_x, ax_y, col in [
        ("Growth<br>CR↑ AOV↑",  0.75, 0.75, "rgba(0,200,100,0.25)"),
        ("Premium<br>CR↓ AOV↑", 0.25, 0.75, "rgba(255,165,0,0.20)"),
        ("Volume<br>CR↑ AOV↓",  0.75, 0.25, "rgba(255,165,0,0.20)"),
        ("Loss<br>CR↓ AOV↓",    0.25, 0.25, "rgba(215,50,50,0.20)"),
    ]:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=ax_x, y=ax_y,
            text=label, showarrow=False,
            font=dict(size=11, color=col),
            opacity=0.7,
        )

    # Arrows from control to each variation
    for i, m in enumerate(metrics_list):
        if i == 0:
            continue
        fig.add_annotation(
            x=m["cr_pct"], y=m["aov"],
            ax=cx, ay=cy,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
            arrowcolor=GROUP_COLORS[i],
            opacity=0.7,
        )

    # Scatter points
    for i, m in enumerate(metrics_list):
        fig.add_trace(go.Scatter(
            x=[m["cr_pct"]], y=[m["aov"]],
            mode="markers+text",
            marker=dict(color=GROUP_COLORS[i], size=14, line=dict(width=1.5, color="#e0e0e0")),
            text=[m["name"]], textposition="top center",
            name=m["name"],
            hovertemplate=f"<b>{m['name']}</b><br>CR: {m['cr_pct']:.2f}%<br>AOV: ${m['aov']:.2f}<extra></extra>",
        ))

    fig.add_vline(x=cx, line=dict(color="#555", dash="dot", width=1))
    fig.add_hline(y=cy, line=dict(color="#555", dash="dot", width=1))

    fig.update_layout(
        title="Strategic Matrix: CR vs AOV",
        xaxis_title="Conversion Rate (%)",
        yaxis_title="Average Order Value ($)",
        plot_bgcolor="#1a1d24",
        paper_bgcolor="#0e1117",
        font_color="#e0e0e0",
        title_font_size=15,
        xaxis=dict(gridcolor="#2e3140"),
        yaxis=dict(gridcolor="#2e3140"),
        showlegend=True,
        margin=dict(t=60, b=50, l=60, r=20),
        hoverlabel=dict(bgcolor="#1a1d24", font_color="#e0e0e0"),
    )
    st.plotly_chart(fig, use_container_width=True)

def _hex_to_rgba(hex_color, alpha=0.12):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def plot_bayesian_pdfs(groups):
    fig = go.Figure()
    for i, g in enumerate(groups):
        a = g["conv"] + 1
        b = max(g["users"] - g["conv"], 0) + 1
        d = beta(a, b)
        x = np.linspace(d.ppf(0.001), d.ppf(0.999), 500)
        y = d.pdf(x)
        col = GROUP_COLORS[i]
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines",
            name=g["name"],
            line=dict(color=col, width=2.5),
            fill="tozeroy",
            fillcolor=_hex_to_rgba(col, 0.12),
            hovertemplate="CR: %{x:.4f}<br>Density: %{y:.2f}<extra>" + g["name"] + "</extra>",
        ))
    fig.update_layout(
        title="Bayesian Posterior Distributions",
        xaxis_title="Conversion Rate",
        yaxis_title="Probability Density",
        plot_bgcolor="#1a1d24",
        paper_bgcolor="#0e1117",
        font_color="#e0e0e0",
        title_font_size=15,
        xaxis=dict(gridcolor="#2e3140"),
        yaxis=dict(gridcolor="#2e3140"),
        legend=dict(bgcolor="#1a1d24", bordercolor="#3a3f52"),
        margin=dict(t=60, b=50, l=60, r=20),
        hoverlabel=dict(bgcolor="#1a1d24", font_color="#e0e0e0"),
    )
    st.plotly_chart(fig, use_container_width=True)

def run_bootstrap_and_plot(uc, cc, uv, cv, alpha_val=0.05, label_v="Variation"):
    rng   = np.random.default_rng()
    pc    = float(np.clip(safe_divide(cc, uc), 0.0, 1.0))
    pv    = float(np.clip(safe_divide(cv, uv), 0.0, 1.0))
    sim_c = rng.binomial(uc, pc, BOOTSTRAP_SAMPLES) / uc
    sim_v = rng.binomial(uv, pv, BOOTSTRAP_SAMPLES) / uv
    diffs = sim_v - sim_c
    ci_l  = float(np.percentile(diffs, alpha_val / 2 * 100))
    ci_h  = float(np.percentile(diffs, (1 - alpha_val / 2) * 100))

    counts, bin_edges = np.histogram(diffs, bins=80)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    conf_label  = f"{(1-alpha_val)*100:.0f}% CI"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_centers, y=counts,
        marker_color="#ff7f0e",
        marker_line_color="#1a1d24",
        marker_line_width=0.5,
        name="Bootstrap samples",
        hovertemplate="Diff: %{x:.4f}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=ci_l, line=dict(color="#E73B37", dash="dash", width=1.8),
                  annotation_text=f"{conf_label} Lower", annotation_font_color="#E73B37",
                  annotation_position="top left")
    fig.add_vline(x=ci_h, line=dict(color="#E73B37", dash="dash", width=1.8),
                  annotation_text=f"{conf_label} Upper", annotation_font_color="#E73B37",
                  annotation_position="top right")
    fig.add_vline(x=0, line=dict(color="#aaaaaa", dash="dot", width=1.2),
                  annotation_text="No effect", annotation_font_color="#aaaaaa",
                  annotation_position="top right")
    fig.update_layout(
        title=f"Bootstrap CR Difference ({label_v} − Control)",
        xaxis_title="Difference in Conversion Rate",
        yaxis_title="Frequency",
        plot_bgcolor="#1a1d24",
        paper_bgcolor="#0e1117",
        font_color="#e0e0e0",
        title_font_size=15,
        bargap=0.02,
        xaxis=dict(gridcolor="#2e3140"),
        yaxis=dict(gridcolor="#2e3140"),
        showlegend=False,
        margin=dict(t=60, b=50, l=60, r=20),
        hoverlabel=dict(bgcolor="#1a1d24", font_color="#e0e0e0"),
    )
    st.plotly_chart(fig, use_container_width=True)
    return sim_c * 100, sim_v * 100, ci_l * 100, ci_h * 100

def plot_box_plots(samples_c, samples_v, label_v="Best Variation"):
    fig = go.Figure()
    for label, samples, color in [
        ("Control", samples_c, GROUP_COLORS[0]),
        (label_v,   samples_v, GROUP_COLORS[1]),
    ]:
        fig.add_trace(go.Box(
            y=samples,
            name=label,
            marker_color=color,
            line_color=color,
            fillcolor=_hex_to_rgba(color, 0.25),
            boxmean=True,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Median: %{median:.3f}%<br>"
                "Q1: %{q1:.3f}%<br>"
                "Q3: %{q3:.3f}%<extra></extra>"
            ),
        ))
    fig.update_layout(
        title="Box Plot: Bootstrap CR Distributions",
        yaxis_title="Conversion Rate (%)",
        plot_bgcolor="#1a1d24",
        paper_bgcolor="#0e1117",
        font_color="#e0e0e0",
        title_font_size=15,
        yaxis=dict(gridcolor="#2e3140"),
        xaxis=dict(showgrid=False),
        showlegend=False,
        margin=dict(t=60, b=40, l=60, r=20),
        hoverlabel=dict(bgcolor="#1a1d24", font_color="#e0e0e0"),
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_power_curve(base_cr, mdes, alpha, max_n):
    sample_sizes = np.linspace(100, max_n, 120)
    fig = go.Figure()

    for i, mde in enumerate(mdes):
        p1 = base_cr / 100.0
        p2 = p1 * (1 + mde / 100.0)
        es = proportion_effectsize(p1, p2)
        powers = [
            NormalIndPower().solve_power(effect_size=es, nobs1=n, alpha=alpha, ratio=1)
            for n in sample_sizes
        ]
        color = GROUP_COLORS[i % len(GROUP_COLORS)]
        fig.add_trace(go.Scatter(
            x=sample_sizes, y=powers,
            mode="lines", name=f"MDE: {mde}%",
            line=dict(color=color, width=2.5),
            hovertemplate=f"MDE {mde}%<br>N: %{{x:,.0f}}<br>Power: %{{y:.2f}}<extra></extra>",
        ))

    fig.add_hline(y=0.80, line=dict(color="#E73B37", dash="dash", width=1.8),
                  annotation_text="80% Power Threshold",
                  annotation_font_color="#E73B37",
                  annotation_position="bottom right")
    fig.update_layout(
        title=f"Statistical Power over Sample Size (Base CR: {base_cr}%)",
        xaxis_title="Sample Size (per group)",
        yaxis_title="Power",
        yaxis_range=[0, 1.05],
        plot_bgcolor="#1a1d24",
        paper_bgcolor="#0e1117",
        font_color="#e0e0e0",
        title_font_size=15,
        xaxis=dict(gridcolor="#2e3140", tickformat=","),
        yaxis=dict(gridcolor="#2e3140"),
        legend=dict(bgcolor="#1a1d24", bordercolor="#3a3f52"),
        margin=dict(t=60, b=50, l=60, r=20),
        hoverlabel=dict(bgcolor="#1a1d24", font_color="#e0e0e0"),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_segment_bars(title, segments, metric_key, label_v="Variation", unit=""):
    """Grouped bar chart: Control vs Variation for each segment."""
    names = [s["name"] for s in segments]
    if metric_key == "cr":
        ctrl_vals = [s["ctrl_cr"] * 100 for s in segments]
        var_vals  = [s["var_cr"]  * 100 for s in segments]
        fmt = ".2f"
        suffix = "%"
    else:
        ctrl_vals = [s["ctrl_rpv"] for s in segments]
        var_vals  = [s["var_rpv"]  for s in segments]
        fmt = ".4f"
        suffix = ""

    var_colors = [GROUP_COLORS[1] if v >= c else "#d62728" for v, c in zip(var_vals, ctrl_vals)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Control",
        x=names, y=ctrl_vals,
        marker_color=GROUP_COLORS[0],
        hovertemplate=f"Control<br>%{{x}}: %{{y:{fmt}}}{suffix}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name=label_v,
        x=names, y=var_vals,
        marker_color=var_colors,
        hovertemplate=f"{label_v}<br>%{{x}}: %{{y:{fmt}}}{suffix}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        yaxis_title=unit,
        barmode="group",
        plot_bgcolor="#1a1d24",
        paper_bgcolor="#0e1117",
        font_color="#e0e0e0",
        title_font_size=15,
        yaxis=dict(gridcolor="#2e3140"),
        xaxis=dict(showgrid=False),
        legend=dict(bgcolor="#1a1d24", bordercolor="#3a3f52"),
        margin=dict(t=60, b=40, l=60, r=20),
        hoverlabel=dict(bgcolor="#1a1d24", font_color="#e0e0e0"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# SIDEBAR
# ============================================================
# ── 1. EXPERIMENT PLANNING ──────────────────────────────────────────────────
st.sidebar.markdown(
    f'<div style="display:flex;align-items:center;">{ICON_CALENDAR}'
    '<h2 style="display:inline;font-size:1.5rem;margin-left:5px;">Experiment Planning</h2></div>',
    unsafe_allow_html=True,
)
with st.sidebar.expander("Sample Size Calculator", expanded=False):
    st.caption("Plan required traffic before you start the test.")
    plan_traffic  = st.number_input("Traffic (Last 28 Days)", step=1000,  key="p_traffic")
    plan_base_cr  = st.number_input("Baseline CR (%)",        step=0.1,   key="p_base_cr")
    plan_base_aov = st.number_input("Baseline AOV ($)",       step=1.0,   key="p_base_aov")
    plan_mde      = st.number_input("Min Detectable Effect (%)", step=0.5, key="p_mde")
    volatility    = st.selectbox("Revenue Variance",
        ["Low (Subscription)", "Medium (Standard E-com)", "High (Whales/B2B)"],
        index=1, key="p_vol")
    sd_mult = {"Low": 1.0, "Medium": 2.0, "High": 3.0}[volatility.split()[0]]
    _conf_for_plan = st.session_state.get("conf_level", "95%")
    _alpha_for_plan = {"90%": 0.10, "95%": 0.05, "99%": 0.01}[_conf_for_plan]
    if st.button("Calculate Duration"):
        if plan_base_cr <= 0 or plan_base_aov <= 0 or plan_traffic <= 0:
            st.error("Traffic, Baseline CR, and AOV must all be > 0.")
        else:
            daily   = plan_traffic / 28
            p1      = plan_base_cr / 100
            p2      = p1 * (1 + plan_mde / 100)
            es_cr   = proportion_effectsize(p1, p2)
            n_cr    = NormalIndPower().solve_power(effect_size=es_cr, alpha=_alpha_for_plan, power=0.8, ratio=1)
            est_sd  = plan_base_aov * sd_mult
            es_rpv  = safe_divide(plan_base_aov * plan_mde / 100, est_sd)
            if es_rpv > 0:
                n_rpv     = TTestIndPower().solve_power(effect_size=es_rpv, alpha=_alpha_for_plan, power=0.8, ratio=1)
                n_rpv_vis = safe_divide(n_rpv, p1)
                days_rpv  = safe_divide(n_rpv_vis * 2, daily)
            else:
                n_rpv_vis, days_rpv = 0, 0
            days_cr = safe_divide(n_cr * 2, daily)
            st.markdown("---")
            st.write(f"**Daily Traffic:** {int(daily):,}")
            st.info(f"**CR:** {int(days_cr)} days ({int(n_cr):,} users/group)")
            (st.error if days_rpv > 60 else st.warning)(
                f"**RPV:** {int(days_rpv)} days ({int(n_rpv_vis):,} users/group)")

st.sidebar.markdown("---")

# ── 2. SETTINGS ─────────────────────────────────────────────────────────────
st.sidebar.markdown(
    f'<div style="display:flex;align-items:center;">{ICON_SETTINGS}'
    '<h2 style="display:inline;font-size:1.5rem;margin-left:5px;">Settings</h2></div>',
    unsafe_allow_html=True,
)

days_run = st.sidebar.number_input("Days Test Ran", min_value=1, key="days")
st.sidebar.date_input(
    "Test Start Date (optional)",
    value=st.session_state.get("start_date"),
    key="start_date",
    help="Enables day-of-week bias detection. Leave blank to skip.",
)

confidence_level = st.sidebar.selectbox(
    "Confidence Level", ["95%", "90%", "99%"], index=0, key="conf_level",
    help="90% — Fast decisions  |  95% — Standard  |  99% — High-stakes",
)
alpha = {"90%": 0.10, "95%": 0.05, "99%": 0.01}[confidence_level]

st.sidebar.selectbox(
    "Primary Goal",
    options=["Maximize CR", "Maximize Revenue", "Balanced"],
    key="primary_goal",
    help=(
        "Maximize CR: Winner = highest conversion rate uplift.\n"
        "Maximize Revenue: Winner = highest RPV uplift — a lower-CR variant can win if it earns more per visitor.\n"
        "Balanced: Composite score weighting CR (40%) and RPV (60%)."
    ),
)
primary_goal = st.session_state.get("primary_goal", "Maximize CR")

with st.sidebar.expander("Guardrail Metrics", expanded=False):
    st.caption("Secondary metrics that must not degrade when you ship a winner.")
    num_guardrails = st.selectbox("Number of guardrail metrics", [1, 2, 3], key="num_guardrails")
    _GUARD_LABELS = ["Guardrail 1", "Guardrail 2", "Guardrail 3"]
    _DIR_OPTIONS  = ["lower is better", "higher is better"]
    for _gi in range(int(num_guardrails)):
        st.markdown(f"**{_GUARD_LABELS[_gi]}**")
        st.text_input("Metric name",            key=f"g{_gi}_name")
        st.number_input("Control value",        min_value=0.0, format="%.2f", key=f"g{_gi}_ctrl")
        st.number_input("Variation value",      min_value=0.0, format="%.2f", key=f"g{_gi}_var")
        st.number_input("Max allowed change (%)", min_value=0.1, step=0.5, key=f"g{_gi}_threshold")
        st.selectbox("Direction", _DIR_OPTIONS, key=f"g{_gi}_dir")
        if _gi < int(num_guardrails) - 1:
            st.markdown("---")

with st.sidebar.expander("Segment Breakdown", expanded=False):
    st.caption("Per-segment CR and RPV to check if the result holds across audiences.")
    num_segments = st.selectbox("Number of segments", [1, 2, 3, 4], key="num_segments")
    for _si in range(int(num_segments)):
        st.markdown(f"**Segment {_si + 1}**")
        st.text_input("Name", key=f"seg{_si}_name")
        st.markdown("Control")
        st.number_input("Users",         min_value=0, key=f"seg{_si}_uc")
        st.number_input("Conversions",   min_value=0, key=f"seg{_si}_cc")
        st.number_input("Revenue ($)",   min_value=0.0, format="%.2f", key=f"seg{_si}_rc")
        st.markdown("Best Variation")
        st.number_input("Users",         min_value=0, key=f"seg{_si}_uv")
        st.number_input("Conversions",   min_value=0, key=f"seg{_si}_cv")
        st.number_input("Revenue ($)",   min_value=0.0, format="%.2f", key=f"seg{_si}_rv")
        if _si < int(num_segments) - 1:
            st.markdown("---")

if int(st.session_state.get("num_variations", 1)) > 1:
    st.sidebar.selectbox(
        "Multiple Comparison Correction",
        options=["holm", "bonferroni", "fdr_bh"],
        format_func=lambda x: {
            "holm":       "Holm-Bonferroni (recommended)",
            "bonferroni": "Bonferroni (most conservative)",
            "fdr_bh":     "Benjamini-Hochberg (FDR)",
        }[x],
        key="mc_method",
        help=(
            "Holm: Controls FWER, more powerful than Bonferroni. Best for most tests.\n"
            "Bonferroni: Most conservative — use when false positives are very costly.\n"
            "FDR (B-H): Controls false discovery rate — best for exploratory multi-variant tests."
        ),
    )
mc_method = st.session_state.get("mc_method", "holm")

st.sidebar.markdown("---")

# ── 3. ENTER RESULTS ────────────────────────────────────────────────────────
st.sidebar.markdown(
    f'<div style="display:flex;align-items:center;">{ICON_BAR_CHART}'
    '<h2 style="display:inline;font-size:1.5rem;margin-left:5px;">Enter Results</h2></div>',
    unsafe_allow_html=True,
)

num_variations = st.sidebar.number_input(
    "Number of Variation Groups", min_value=1, max_value=MAX_VARIATIONS, key="num_variations",
    help="1 = classic A/B. Add up to 3 variations for A/B/n testing.",
)

st.sidebar.subheader("Control Group")
users_c = st.sidebar.number_input("Users",         min_value=1,   key="users_c")
conv_c  = st.sidebar.number_input("Conversions",   min_value=0,   key="conv_c")
rev_c   = st.sidebar.number_input("Revenue ($)",   min_value=0.0, key="rev_c")
prod_c  = st.sidebar.number_input("Products Sold", min_value=0,   key="prod_c")

VAR_LABELS = ["Variation A", "Variation B", "Variation C"]
var_inputs = []
for i in range(int(num_variations)):
    st.sidebar.markdown("---")
    st.sidebar.subheader(VAR_LABELS[i])
    var_inputs.append({
        "name":  VAR_LABELS[i],
        "users": st.sidebar.number_input("Users",         min_value=1,   key=f"users_v{i}"),
        "conv":  st.sidebar.number_input("Conversions",   min_value=0,   key=f"conv_v{i}"),
        "rev":   st.sidebar.number_input("Revenue ($)",   min_value=0.0, key=f"rev_v{i}"),
        "prod":  st.sidebar.number_input("Products Sold", min_value=0,   key=f"prod_v{i}"),
    })

st.sidebar.markdown("---")

# ── 4. SAVE & LOAD ───────────────────────────────────────────────────────────
st.sidebar.markdown(
    f'<div style="display:flex;align-items:center;">{ICON_UPLOAD}'
    '<h2 style="display:inline;font-size:1.5rem;margin-left:5px;">Save & Load Analysis</h2></div>',
    unsafe_allow_html=True,
)
st.sidebar.info("Snapshot your experiment. Download Day 7, re-upload Day 14 to track evolution.")
snapshot = {k: st.session_state.get(k) for k in SAVE_KEYS}
st.sidebar.download_button("Download Inputs (.json)", json.dumps(snapshot, indent=2),
                   "experiment_snapshot.json", "application/json")
_pdf_requested = st.sidebar.button("Prepare PDF Report", key="_pdf_btn",
                                    help="Builds a full PDF report with tables, charts, and analysis.")
uploaded = st.sidebar.file_uploader("Load Snapshot", type=["json"])
if uploaded is not None:
    try:
        loaded = json.load(uploaded)
        st.session_state["_pending_load"] = {k: loaded[k] for k in SAVE_KEYS if k in loaded}
        st.rerun()
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        st.sidebar.error(f"Could not load file: {e}")


# ============================================================
# BUILD GROUPS & RUN ANALYSIS
# ============================================================
groups = [{"name": "Control", "users": users_c, "conv": conv_c, "rev": rev_c, "prod": prod_c}] + var_inputs

# Input validation — warn in sidebar if conversions exceed users for any group
for _g in groups:
    if _g["conv"] > _g["users"]:
        st.sidebar.error(f"⚠ {_g['name']}: Conversions ({_g['conv']:,}) exceed Users ({_g['users']:,}). Conversions will be clamped.")

mv      = run_multivariate_analysis(groups, alpha, mc_method, primary_goal)
bayes   = calculate_bayesian_multivariate(groups)
srm_stat, p_srm = perform_srm_test([g["users"] for g in groups])

ctrl_m  = mv["metrics"][0]
# Prefer the declared winner for bootstrap/revenue analysis;
# fall back to highest-CR variation so the tab always has something to show.
if mv["winner"]:
    best_m = next(m for m in mv["metrics"] if m["name"] == mv["winner"])
elif len(mv["metrics"]) > 1:
    best_m = max(mv["metrics"][1:], key=lambda m: m["cr"])
else:
    best_m = ctrl_m
best_g  = next(g for g in groups if g["name"] == best_m["name"])

rev_sig = test_revenue_significance(
    users_c, conv_c, rev_c,
    best_g["users"], best_g["conv"], best_g["rev"],
    alpha=alpha,
)

_guardrail_inputs = [
    {
        "name":      st.session_state[f"g{i}_name"],
        "ctrl_val":  float(st.session_state[f"g{i}_ctrl"]),
        "var_val":   float(st.session_state[f"g{i}_var"]),
        "threshold": float(st.session_state[f"g{i}_threshold"]),
        "direction": st.session_state[f"g{i}_dir"],
    }
    for i in range(int(st.session_state.get("num_guardrails", 1)))
]
guardrail_results = evaluate_guardrails(_guardrail_inputs)

_segment_inputs = [
    {
        "name": st.session_state[f"seg{i}_name"],
        "uc":   int(st.session_state[f"seg{i}_uc"]),
        "cc":   int(st.session_state[f"seg{i}_cc"]),
        "rc":   float(st.session_state[f"seg{i}_rc"]),
        "uv":   int(st.session_state[f"seg{i}_uv"]),
        "cv":   int(st.session_state[f"seg{i}_cv"]),
        "rv":   float(st.session_state[f"seg{i}_rv"]),
    }
    for i in range(int(st.session_state.get("num_segments", 1)))
]
segment_results = analyze_segments(_segment_inputs, alpha)

duration_checks   = analyze_test_duration(days_run, st.session_state.get("start_date"))

# ---- PDF generation (triggered by sidebar button, runs after all data is ready) ----
if _pdf_requested:
    st.session_state["_pdf_bytes"] = generate_pdf_report(
        mv, bayes, rev_sig, guardrail_results, duration_checks,
        p_srm, groups, days_run, confidence_level, primary_goal,
        ctrl_m, best_m,
        hypothesis=st.session_state.get("hyp_smart", ""),
        smart_text=st.session_state.get("_smart_report_text", ""),
        ai_text=st.session_state.get("_ai_report_text", ""),
        segment_results=segment_results,
    )
if st.session_state.get("_pdf_bytes"):
    st.sidebar.download_button(
        "Download PDF Report",
        data=st.session_state["_pdf_bytes"],
        file_name=f"ab_test_report_{datetime.date.today().isoformat()}.pdf",
        mime="application/pdf",
        key="_pdf_dl",
    )

# ---- Pre-compute bootstrap samples at top level so both Tab 9 & Tab 10
#      can share the same data without cross-tab session-state coupling. ----
_boot_rng   = np.random.default_rng()
_boot_pc    = float(np.clip(safe_divide(ctrl_m["conv"], ctrl_m["users"]), 0.0, 1.0))
_boot_pv    = float(np.clip(safe_divide(best_m["conv"], best_m["users"]), 0.0, 1.0))
_boot_sc    = _boot_rng.binomial(ctrl_m["users"], _boot_pc, BOOTSTRAP_SAMPLES) / ctrl_m["users"] * 100
_boot_sv    = _boot_rng.binomial(best_m["users"], _boot_pv, BOOTSTRAP_SAMPLES) / best_m["users"] * 100


# ============================================================
# MAIN DASHBOARD
# ============================================================
st.title("Enterprise A/B Test Analyzer")
render_header(ICON_BAR_CHART, "Results Summary")

# Omnibus banner (only meaningful for multi-variant)
if num_variations > 1:
    if mv["p_global"] <= alpha:
        st.success(
            f"🔬 **Omnibus χ² Test:** At least one variant differs significantly "
            f"(p = {mv['p_global']:.4f}). Pairwise results use **{mc_method.upper()}** correction."
        )
    else:
        st.info(f"🔬 **Omnibus χ² Test:** No significant overall difference (p = {mv['p_global']:.4f}).")

# --- KPI Table ---
st.subheader("1. Primary KPIs")
kpi_rows = []
for m in mv["metrics"]:
    pw = next((p for p in mv["pairwise"] if p["name"] == m["name"]), None)
    kpi_rows.append({
        "Group":       m["name"],
        "Users":       f"{m['users']:,}",
        "Conv Rate":   f"{m['cr_pct']:.2f}%",
        "CR Uplift":   f"{m['uplift_cr']:+.2f}%" if pw else "—",
        "RPV":         f"${m['rpv']:.2f}",
        "RPV Uplift":  f"{m['uplift_rpv']:+.2f}%" if pw else "—",
        "AOV":         f"${m['aov']:.2f}",
        "p (adj)":     (f"{pw['p_adjusted']:.4f}" if np.isfinite(pw['p_adjusted']) else "—") if pw else "—",
        "Significant": ("✅" if pw["significant"] else "❌") if pw else "—",
    })
st.dataframe(pd.DataFrame(kpi_rows), use_container_width=True, hide_index=True)

# Winner callout
if mv["winner"]:
    st.markdown(
        f'<div style="background:#1a3d1a;border-left:4px solid #2ca02c;padding:10px 16px;border-radius:4px;">'
        f'{ICON_TROPHY}<strong>Winner: {mv["winner"]}</strong> — best overall performance after {mc_method.upper()} correction (CR → RPV → AOV → product efficiency).</div>',
        unsafe_allow_html=True,
    )
    if mv.get("winner_rpv_negative"):
        st.warning(
            f"⚠️ **Volume Play Warning:** {mv['winner']} has a negative RPV uplift. "
            "CR is up but revenue per visitor is down — review revenue impact before shipping."
        )
else:
    st.warning("No variant has achieved a statistically significant positive uplift yet.")

# --- Product Velocity ---
st.subheader("2. Product Velocity")
pv_rows = []
for m in mv["metrics"]:
    apo_up = safe_divide((m["apo"] - ctrl_m["apo"]) * 100, ctrl_m["apo"]) if m["name"] != "Control" else None
    pv_rows.append({
        "Group": m["name"],
        "Avg Products / Order": f"{m['apo']:.2f}",
        "Avg Products / User":  f"{m['apu']:.2f}",
        "APO Uplift":           f"{apo_up:+.2f}%" if apo_up is not None else "—",
    })
st.dataframe(pd.DataFrame(pv_rows), use_container_width=True, hide_index=True)

st.markdown("---")

# --- Executive Interpretation ---
st.subheader("3. Executive Interpretation")
lead_pw = mv["pairwise"][0] if mv["pairwise"] else None
if lead_pw:
    if lead_pw["uplift_cr"] > 0 and lead_pw["uplift_aov"] < 0:
        st.warning("Trade-off: Leading variant drives MORE conversions but SMALLER baskets.")
    elif lead_pw["uplift_cr"] < 0 and lead_pw["uplift_aov"] > 0:
        st.warning("Trade-off: Leading variant drives FEWER conversions but BIGGER baskets.")
    else:
        st.success("Clean Result: CR and AOV trends are consistent for the leading variant.")

rpv_delta = best_m["rpv"] - ctrl_m["rpv"]
direction = "more" if rpv_delta >= 0 else "less"
st.write(f"**Financial Impact ({best_m['name']}):** ${abs(rpv_delta):.2f} {direction} per visitor.")

# Revenue significance expander
with st.expander(f"Revenue Significance — Control vs {best_m['name']}", expanded=False):
    st.caption(
        "Revenue is skewed by large orders — z-tests are unreliable. "
        "Uses Mann-Whitney U and log-transformed bootstrap. "
        "Distributions are reconstructed from aggregates — treat as directional signals."
    )
    st.markdown("---")
    rc1, rc2 = st.columns(2)
    for col, label, key, delta in [
        (rc1, "Revenue Per Visitor (RPV)", "rpv", best_m["rpv"] - ctrl_m["rpv"]),
        (rc2, "Average Order Value (AOV)",  "aov", best_m["aov"] - ctrl_m["aov"]),
    ]:
        with col:
            st.markdown(f"#### {label}")
            res = rev_sig[key]
            ma, mb = st.columns(2)
            ma.metric("Observed Δ", f"${delta:+.3f}")
            mb.metric("MW p-value", f"{res['mw_p']:.4f}")
            if res["mw_sig"]:
                st.success(f"✅ Mann-Whitney: Significant at {alpha*100:.0f}%")
            else:
                st.info(f"Mann-Whitney: Not significant (p={res['mw_p']:.4f})")
            if res["boot_sig"]:
                st.success(f"✅ Bootstrap: Significant (p={res['boot_p']:.4f})")
            else:
                st.info(f"Bootstrap: Not significant (p={res['boot_p']:.4f})")
            cl, ch = res["boot_ci_low"], res["boot_ci_high"]
            colour = "green" if cl > 0 else ("red" if ch < 0 else "orange")
            st.markdown(
                f"**{int((1-alpha)*100)}% CI on Δ:** "
                f"<span style='color:{colour}'>${cl:.3f} to ${ch:.3f}</span>",
                unsafe_allow_html=True,
            )
            if cl > 0:   st.success("CI entirely positive — gain is consistent.")
            elif ch < 0: st.error("CI entirely negative — loss is consistent.")
            else:        st.warning("CI crosses zero — result is uncertain.")

# --- Health Checks ---
st.subheader("4. Health Checks")
hc1, hc2, hc3 = st.columns(3)

with hc1:
    if p_srm < 0.01:
        st.error(f"SRM DETECTED (p={p_srm:.4f}) — traffic split uneven. Results may be invalid.")
    else:
        st.success(f"SRM PASSED (p={p_srm:.4f})")

_RENDER = {"error": st.error, "warning": st.warning, "pass": st.success}

with hc2:
    _dur_ids = {"too_short", "novelty", "duration_ok"}
    for _chk in duration_checks:
        if _chk["id"] in _dur_ids:
            _RENDER[_chk["level"]](f"**{_chk['label']}:** {_chk['msg']}")

with hc3:
    _biz_ids = {"incomplete_weeks", "weeks_ok", "long_running", "dow_bias", "dow_ok"}
    _biz_checks = [c for c in duration_checks if c["id"] in _biz_ids]
    if not _biz_checks:
        st.success("Business cycle: No issues detected.")
    else:
        for _chk in _biz_checks:
            _RENDER[_chk["level"]](f"**{_chk['label']}:** {_chk['msg']}")

st.markdown("---")

# --- Guardrail Metrics ---
st.subheader("5. Guardrail Metrics")
st.caption("Secondary metrics that must not degrade when shipping a winner. Configure thresholds in the sidebar.")

_SVG_PASS = """<svg viewBox="0 0 1024 1024" style="width:1em;height:1em;vertical-align:middle;margin-right:6px;" xmlns="http://www.w3.org/2000/svg"><path d="M512 64C264 64 64 264 64 512s200 448 448 448 448-200 448-448S760 64 512 64z m0 856C291 920 108 737 108 512S291 104 512 104s404 183 404 404-183 404-404 404z" fill="#2ca02c"/><path d="M420 680l-160-160 31-31 129 129 269-269 31 31z" fill="#2ca02c"/></svg>"""
_SVG_FAIL = """<svg viewBox="0 0 1024 1024" style="width:1em;height:1em;vertical-align:middle;margin-right:6px;" xmlns="http://www.w3.org/2000/svg"><path d="M512 64C264 64 64 264 64 512s200 448 448 448 448-200 448-448S760 64 512 64z m0 856C291 920 108 737 108 512S291 104 512 104s404 183 404 404-183 404-404 404z" fill="#E73B37"/><path d="M676 320l-164 164-164-164-28 28 164 164-164 164 28 28 164-164 164 164 28-28-164-164 164-164z" fill="#E73B37"/></svg>"""

_active_guardrails = [r for r in guardrail_results if not r["skip"]]
if _active_guardrails:
    _any_violated = any(r["violated"] for r in _active_guardrails)
    if _any_violated:
        st.markdown(
            f'<div style="background:#3d1a1a;border-left:4px solid #E73B37;padding:10px 16px;border-radius:4px;margin-bottom:12px;">'
            f'{_SVG_FAIL}<span style="color:#E73B37;font-weight:600;">One or more guardrail metrics are violated — review before shipping.</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="background:#1a3d1a;border-left:4px solid #2ca02c;padding:10px 16px;border-radius:4px;margin-bottom:12px;">'
            f'{_SVG_PASS}<span style="color:#2ca02c;font-weight:600;">All guardrail metrics are within acceptable thresholds.</span></div>',
            unsafe_allow_html=True,
        )
    g_cols = st.columns(len(guardrail_results))
    for col, r in zip(g_cols, guardrail_results):
        with col:
            if r["skip"]:
                st.info(f"**{r['name']}**\n\nNo control value entered.")
            else:
                arrow  = "▲" if r["delta_pct"] > 0 else "▼"
                st.metric(
                    label=r["name"],
                    value=f"{r['var']:.2f}",
                    delta=f"{arrow} {abs(r['delta_pct']):.1f}% (limit ±{r['threshold']}%)",
                    delta_color="inverse" if r["direction"] == "lower is better" else "normal",
                )
                if r["violated"]:
                    st.markdown(
                        f'<div style="display:flex;align-items:center;margin-top:4px;">'
                        f'{_SVG_FAIL}<span style="color:#E73B37;font-size:0.85em;">Exceeds {r["threshold"]}% threshold ({r["direction"]})</span></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="display:flex;align-items:center;margin-top:4px;">'
                        f'{_SVG_PASS}<span style="color:#2ca02c;font-size:0.85em;">Within {r["threshold"]}% threshold ({r["direction"]})</span></div>',
                        unsafe_allow_html=True,
                    )
else:
    st.info("Enter guardrail metric values in the sidebar to enable this check.")

st.markdown("---")

# ============================================================
# SIMPSON'S PARADOX
# ============================================================
render_header(ICON_PIE, "Advanced: Simpson's Paradox Detector", level=3)
with st.expander("Expand to check segments (e.g. Mobile vs Desktop)", expanded=False):
    st.caption("Detects when segment-level trends contradict the aggregate result.")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.subheader("Segment 1")
        s1_uc = st.number_input("Control Users",  min_value=0, key="s1_uc")
        s1_cc = st.number_input("Control Conv.",  min_value=0, key="s1_cc")
        s1_uv = st.number_input("Var Users",      min_value=0, key="s1_uv")
        s1_cv = st.number_input("Var Conv.",       min_value=0, key="s1_cv")
    with sc2:
        st.subheader("Segment 2")
        s2_uc = st.number_input("Control Users",  min_value=0, key="s2_uc")
        s2_cc = st.number_input("Control Conv.",  min_value=0, key="s2_cc")
        s2_uv = st.number_input("Var Users",      min_value=0, key="s2_uv")
        s2_cv = st.number_input("Var Conv.",       min_value=0, key="s2_cv")
    if st.button("Check for Paradox"):
        paradox, up1, up2, up_agg = check_simpsons_paradox(
            {"users_c": s1_uc, "conv_c": s1_cc, "users_v": s1_uv, "conv_v": s1_cv},
            {"users_c": s2_uc, "conv_c": s2_cc, "users_v": s2_uv, "conv_v": s2_cv},
        )
        st.markdown("---")
        pm1, pm2, pm3 = st.columns(3)
        pm1.metric("Segment 1 Uplift", f"{up1:.2f}%")
        pm2.metric("Segment 2 Uplift", f"{up2:.2f}%")
        pm3.metric("Combined Uplift",  f"{up_agg:.2f}%")
        if paradox:
            st.error("🚨 SIMPSON'S PARADOX DETECTED!")
            st.warning("Trust segment data — aggregate result is misleading due to imbalance.")
        else:
            st.success("✅ No paradox. Trends are consistent across segments.")

st.markdown("---")

# ============================================================
# DEEP DIVE TABS
# ============================================================
render_header(ICON_BRAIN, "Deep Dive Analysis")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
    "Smart Analysis", "AI Analysis", "Stopping & Sequential",
    "Strategic Matrix", "Product Metrics", "Revenue Charts",
    "CR Comparison", "Bayesian", "Bootstrap", "Box Plot", "Power Curves",
    "Segment Breakdown"
])

# ---- TAB 1: SMART ANALYSIS ----
with tab1:
    st.markdown("### Auto-Analyst Report")
    st.info("Instant rule-based analysis — no API key required.")
    hyp_smart = st.text_area("Hypothesis:", placeholder="We believed that...", height=70, key="hyp_smart")
    if st.button("Generate Smart Report"):
        smart_payload = {
            "days":             days_run,
            "p_srm":            p_srm,
            "rev_sig":          rev_sig,
            "guardrail_results": guardrail_results,
            "duration_checks":  duration_checks,
            "primary_goal":     primary_goal,
            "metrics":          mv["metrics"],
        }
        _smart_text = generate_smart_analysis(hyp_smart, mv, bayes, smart_payload, alpha, segment_results=segment_results)
        st.session_state["_smart_report_text"] = _smart_text
        st.markdown("---")
        st.markdown(_smart_text)

# ---- TAB 2: AI ANALYSIS ----
with tab2:
    st.markdown("### AI-Powered Analysis")
    col_prov, col_key = st.columns(2)
    with col_prov:
        ai_provider = st.selectbox("Provider", ["OpenAI (GPT-4o)", "DeepSeek (R1)"])
    with col_key:
        api_key_input = st.text_input("API Key", type="password")
    hyp_ai = st.text_area("Hypothesis:", placeholder="We believed that...", height=100, key="hyp_ai")
    if st.button("Generate AI Analysis"):
        ai_payload = {
            "num_variations": int(num_variations),
            "correction_method": mc_method,
            "days": days_run,
            "p_srm": p_srm,
            "omnibus_p": mv["p_global"],
            "winner": mv["winner"],
            "pairwise_results": mv["pairwise"],
            # Include all computed rates — the AI needs cr/aov/rpv/apo/apu for its analysis
            "per_group_metrics": mv["metrics"],
            "bayesian_prob_best":   bayes["prob_best"],
            "bayesian_exp_loss":    bayes["expected_loss"],
            "rpv_sig":  rev_sig["rpv"]["sig"],  "rpv_mw_p":  rev_sig["rpv"]["mw_p"],
            "rpv_boot_ci": [rev_sig["rpv"]["boot_ci_low"], rev_sig["rpv"]["boot_ci_high"]],
            "aov_sig":  rev_sig["aov"]["sig"],  "aov_mw_p":  rev_sig["aov"]["mw_p"],
            "aov_boot_ci": [rev_sig["aov"]["boot_ci_low"], rev_sig["aov"]["boot_ci_high"]],
        }
        provider_name = "DeepSeek" if "DeepSeek" in ai_provider else "OpenAI"
        with st.spinner(f"Connecting to {provider_name}..."):
            ai_result = get_ai_analysis(
                api_key_input, hyp_ai, ai_payload,
                provider=provider_name, conf_level=confidence_level,
                segment_results=segment_results,
            )
        st.session_state["_ai_report_text"] = ai_result
        st.markdown("---")
        st.markdown(ai_result)

# ---- TAB 3: STOPPING & SEQUENTIAL ----
with tab3:
    st.markdown("### Bayesian Stopping Rules")
    prob_cols = st.columns(len(groups))
    for i, g in enumerate(groups):
        prob  = bayes["prob_best"].get(g["name"], 0)
        loss  = bayes["expected_loss"].get(g["name"], 0)
        prob_cols[i].metric(f"{g['name']} — P(Best)", f"{prob*100:.1f}%")
        prob_cols[i].caption(f"Expected loss: {loss*100:.5f}%")

    st.markdown("---")
    st.subheader("Sequential Testing / Peeking Penalty")
    st.caption(
        "Repeated checks inflate the false positive rate. "
        "This log-based approximation adjusts your alpha threshold. "
        "For production use, consider O'Brien-Fleming or Pocock alpha-spending."
    )
    peeks = st.number_input("How many times have you checked results?", min_value=1, value=1)
    adj_alpha = alpha * np.log(1 + (np.e - 1) / peeks)
    st.write(f"Standard: **{alpha}** → Adjusted: **{adj_alpha:.5f}**")
    for pw in mv["pairwise"]:
        if pw["p_raw"] < adj_alpha:
            st.success(f"✅ {pw['name']}: Significant after penalty (p={pw['p_raw']:.4f})")
        else:
            st.error(f"❌ {pw['name']}: Not significant after penalty (p={pw['p_raw']:.4f})")

# ---- TAB 4: STRATEGIC MATRIX ----
with tab4:
    plot_strategic_matrix(mv["metrics"])

# ---- TAB 5: PRODUCT METRICS ----
with tab5:
    labels = [m["name"] for m in mv["metrics"]]
    c1, c2 = st.columns(2)
    with c1:
        plot_multivariant_bar("Avg Products / Order", [m["apo"] for m in mv["metrics"]], labels)
    with c2:
        plot_multivariant_bar("Avg Products / User",  [m["apu"] for m in mv["metrics"]], labels)

# ---- TAB 6: REVENUE CHARTS ----
with tab6:
    labels = [m["name"] for m in mv["metrics"]]
    c1, c2 = st.columns(2)
    with c1:
        plot_multivariant_bar("Revenue Per Visitor", [m["rpv"] for m in mv["metrics"]], labels, "$")
    with c2:
        plot_multivariant_bar("Avg Order Value",     [m["aov"] for m in mv["metrics"]], labels, "$")

# ---- TAB 7: CR COMPARISON ----
with tab7:
    labels = [m["name"] for m in mv["metrics"]]
    plot_multivariant_bar("Conversion Rate (%)", [m["cr_pct"] for m in mv["metrics"]], labels)

# ---- TAB 8: BAYESIAN PDFs ----
with tab8:
    plot_bayesian_pdfs(groups)

# ---- TAB 9 & 10: BOOTSTRAP + BOX PLOT (Control vs best variation) ----
# Both tabs use pre-computed top-level bootstrap samples (_boot_sc / _boot_sv)
# so visiting Tab 10 never requires visiting Tab 9 first.
with tab9:
    st.markdown(f"### Bootstrap CI — Control vs {best_m['name']}")
    samps_c, samps_v, bci_l, bci_h = run_bootstrap_and_plot(
        ctrl_m["users"], ctrl_m["conv"],
        best_m["users"], best_m["conv"],
        alpha_val=alpha, label_v=best_m["name"],
    )
    st.write(
        f"**{confidence_level} CI on CR Difference ({best_m['name']} − Control):** "
        f"{bci_l:.2f}% to {bci_h:.2f}%"
    )

with tab10:
    st.markdown("### Box Plot: Bootstrap Distributions")
    plot_box_plots(_boot_sc, _boot_sv, label_v=best_m["name"])

# ---- TAB 11: POWER CURVES ----
with tab11:
    st.markdown("### Minimum Detectable Effect (MDE) & Statistical Power")
    st.info("Visualise how sample size and expected business impact (MDE) affect the probability of detecting a real uplift.")
    
    pc_col1, pc_col2 = st.columns(2)
    with pc_col1:
        base_cr_pc = st.number_input("Baseline CR (%)", value=st.session_state.get("p_base_cr", 2.5), step=0.1, key="pc_base_cr")
        max_Traffic = plan_traffic if 'plan_traffic' in locals() and plan_traffic > 1000 else 50000
        max_sample = st.number_input("Max Sample Size (per group)", min_value=1000, max_value=5000000, value=int(max_Traffic), step=1000, key="pc_max_n")
    with pc_col2:
        mde_input = st.text_input("MDEs to plot (%, comma separated)", value="2, 5, 10", key="pc_mdes")
        
    try:
        mdes = [float(x.strip()) for x in mde_input.split(",") if x.strip()]
        if not mdes:
            st.warning("Please enter at least one valid MDE percentage.")
        elif base_cr_pc <= 0:
            st.warning("Baseline CR must be greater than 0.")
        else:
            plot_power_curve(base_cr_pc, mdes, alpha, max_sample)
    except ValueError:
        st.error("Invalid MDE format. Please use comma-separated numbers (e.g., 2, 5.5, 10).")

# ---- TAB 12: SEGMENT BREAKDOWN ----
with tab12:
    st.markdown("### Segment Breakdown Explorer")
    st.caption(
        f"Per-segment CR and RPV for Control vs **{best_m['name']}**. "
        "Configure segments in the sidebar under Settings > Segment Breakdown."
    )
    st.caption(
        "Significance uses raw p-values — no multiple comparison correction applied. "
        "Treat this as an exploratory consistency check, not primary evidence."
    )

    active_segs = [s for s in segment_results if not s["skip"]]

    if not active_segs:
        st.info("No segment data entered yet. Add values in the sidebar under Settings > Segment Breakdown.")
    else:
        # ── Consistency check banner ──
        wins  = sum(1 for s in active_segs if s["uplift_cr"] > 0)
        total = len(active_segs)
        if wins > total / 2:
            st.success(
                f"Variation leads in **{wins}/{total}** segments — result is consistent across audiences."
            )
        elif wins == total / 2:
            st.warning(
                f"Variation leads in **{wins}/{total}** segments — mixed result. "
                "Check RPV and significance before deciding."
            )
        else:
            st.warning(
                f"Variation leads in only **{wins}/{total}** segments — "
                "aggregate uplift may not generalise across audiences."
            )

        # ── Segment table ──
        import pandas as _pd
        _rows = []
        for s in active_segs:
            _rows.append({
                "Segment":    s["name"],
                "Ctrl CR":    f"{s['ctrl_cr']*100:.2f}%",
                "Var CR":     f"{s['var_cr']*100:.2f}%",
                "CR Uplift":  f"{s['uplift_cr']:+.2f}%",
                "Ctrl RPV":   f"${s['ctrl_rpv']:.4f}",
                "Var RPV":    f"${s['var_rpv']:.4f}",
                "RPV Uplift": f"{s['uplift_rpv']:+.2f}%",
                "p-value":    f"{s['p_value']:.4f}",
                "Sig":        "Yes" if s["significant"] else "No",
            })
        st.dataframe(_pd.DataFrame(_rows), use_container_width=True, hide_index=True)

        # ── Charts ──
        _has_rev = any(s["ctrl_rpv"] > 0 or s["var_rpv"] > 0 for s in active_segs)
        if _has_rev:
            _col_cr, _col_rpv = st.columns(2)
            with _col_cr:
                plot_segment_bars("Conversion Rate", active_segs, "cr", label_v=best_m["name"], unit="CR (%)")
            with _col_rpv:
                plot_segment_bars("Revenue Per Visitor", active_segs, "rpv", label_v=best_m["name"], unit="RPV ($)")
        else:
            plot_segment_bars("Conversion Rate", active_segs, "cr", label_v=best_m["name"], unit="CR (%)")
