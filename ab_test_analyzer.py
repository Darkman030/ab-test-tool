import streamlit as st
import numpy as np
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # Must be set before importing pyplot
import matplotlib.pyplot as plt
from scipy.stats import beta, chisquare, mannwhitneyu, chi2_contingency
from statsmodels.stats.proportion import proportions_ztest, proportion_confint, proportion_effectsize
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.multitest import multipletests
import openai

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
        "days": 14, "conf_level": "95%", "mc_method": "holm",
        "p_traffic": 50000, "p_base_cr": 2.5, "p_base_aov": 75.0,
        "p_mde": 5.0, "p_vol": "Medium (Standard E-com)",
        "s1_uc": 2000, "s1_cc": 100, "s1_uv": 2000, "s1_cv": 110,
        "s2_uc": 3000, "s2_cc": 400, "s2_uv": 3000, "s2_cv": 420,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

initialize_state()

SAVE_KEYS = [
    "num_variations",
    "users_c","conv_c","rev_c","prod_c",
    "users_v0","conv_v0","rev_v0","prod_v0",
    "users_v1","conv_v1","rev_v1","prod_v1",
    "users_v2","conv_v2","rev_v2","prod_v2",
    "days","conf_level","mc_method",
    # Sample-size calculator inputs (preserved in snapshots)
    "p_traffic","p_base_cr","p_base_aov","p_mde","p_vol",
    "s1_uc","s1_cc","s1_uv","s1_cv",
    "s2_uc","s2_cc","s2_uv","s2_cv",
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
# MULTI-VARIANT STATISTICAL ENGINE
# -----------------------------------------------
def run_multivariate_analysis(groups, alpha, mc_method):
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
        cr  = safe_divide(g["conv"],  g["users"])
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
    if len(raw_p) > 0:
        reject, p_adj, _, _ = multipletests(raw_p, alpha=alpha, method=mc_method)
    else:
        reject, p_adj = np.array([]), np.array([])

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
        })

    # --- Winner = significant variation with positive CR uplift AND non-negative RPV ---
    # Requiring uplift_rpv >= 0 prevents declaring a winner that costs net revenue.
    sig_winners = [
        pw for pw in pairwise
        if pw["significant"] and pw["uplift_cr"] > 0
    ]
    # Separate clean wins (CR up, RPV up) from volume-only wins (CR up, RPV down)
    clean_winners = [pw for pw in sig_winners if pw["uplift_rpv"] >= 0]
    winner = (
        max(clean_winners, key=lambda x: x["uplift_cr"])["name"]
        if clean_winners
        else (max(sig_winners, key=lambda x: x["uplift_cr"])["name"] if sig_winners else None)
    )
    # Flag if winner was chosen despite negative RPV (volume play)
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
def get_ai_analysis(api_key, hypothesis, metrics, provider="OpenAI", conf_level="95%"):
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
{json.dumps(metrics, indent=2)}

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
def generate_smart_analysis(hypothesis, mv_results, bayes_mv, metrics_payload, alpha_val):
    report = []

    if metrics_payload["p_srm"] < 0.01:
        report.append("### ⛔ CRITICAL: Sample Ratio Mismatch Detected")
        report.append(
            f"SRM p = **{metrics_payload['p_srm']:.4f}**. Traffic split is uneven — "
            "**all results below are likely invalid.** Fix your randomisation before drawing conclusions."
        )
        return "\n\n".join(report)

    n_vars  = mv_results["n_comparisons"]
    winner  = mv_results["winner"]
    p_glob  = mv_results["p_global"]

    # Headline
    if n_vars == 1:
        pw = mv_results["pairwise"][0]
        if pw["significant"] and pw["uplift_cr"] > 0:
            headline = "✅ WINNER: Statistically Significant Positive Result"
            summary  = f"Variation outperforms Control at {(1-alpha_val)*100:.0f}% confidence (p_adj = {pw['p_adjusted']:.4f})."
        elif pw["significant"] and pw["uplift_cr"] < 0:
            headline = "❌ LOSER: Statistically Significant Negative Result"
            summary  = f"Variation is significantly worse than Control (p_adj = {pw['p_adjusted']:.4f}). Do not ship."
        else:
            headline = "⚠️ INCONCLUSIVE: No Clear Winner"
            summary  = f"Cannot reject the null hypothesis (p_adj = {pw['p_adjusted']:.4f}). More data needed."
    else:
        if winner:
            headline = f"✅ MULTI-VARIANT WINNER: {winner}"
            summary  = (
                f"Omnibus test confirms a significant difference (χ² p = {p_glob:.4f}). "
                f"**{winner}** is the strongest performer after "
                f"{mv_results['correction'].upper()} correction."
            )
        elif p_glob <= alpha_val:
            headline = "⚠️ SIGNIFICANT DIFFERENCE — No Positive Winner"
            summary  = (
                f"Omnibus test significant (p = {p_glob:.4f}) but no variant shows a "
                "significant positive uplift after correction. One may be significantly worse."
            )
        else:
            headline = "⚠️ MULTI-VARIANT INCONCLUSIVE"
            summary  = f"No significant overall difference (omnibus p = {p_glob:.4f})."

    report.append(f"### {headline}")
    report.append(summary)
    if hypothesis:
        report.append(f"**Hypothesis:** _{hypothesis}_")

    # Data health
    report.append("### Data Health & Validity")
    days = metrics_payload["days"]
    if days < 7:
        report.append(f"- ⚠️ **Duration:** Only {days} days — too short.")
    elif days < 14:
        report.append(f"- ⚠️ **Duration:** {days} days — watch for novelty effects.")
    else:
        report.append(f"- ✅ **Duration:** {days} days — healthy.")
    report.append(f"- ✅ **SRM:** Passed (p = {metrics_payload['p_srm']:.4f}).")
    if n_vars > 1:
        report.append(
            f"- ℹ️ **MCC:** {mv_results['correction'].upper()} applied across "
            f"{n_vars} pairwise comparisons."
        )

    # Pairwise table
    report.append("### Pairwise Results vs Control")
    for pw in mv_results["pairwise"]:
        sig = "✅ Significant" if pw["significant"] else "❌ Not Significant"
        d   = "▲" if pw["uplift_cr"] > 0 else "▼"
        report.append(
            f"- **{pw['name']}**: CR {d}{pw['uplift_cr']:+.2f}% | "
            f"RPV {pw['uplift_rpv']:+.2f}% | "
            f"p_raw={pw['p_raw']:.4f} → p_adj={pw['p_adjusted']:.4f} — {sig}"
        )

    # Bayesian
    report.append("### Bayesian Assessment")
    for name, prob in bayes_mv["prob_best"].items():
        loss = bayes_mv["expected_loss"].get(name, 0)
        report.append(f"- **{name}**: {prob*100:.1f}% P(best) | Expected loss: {loss*100:.5f}%")

    # Conclusion
    report.append("### Strategic Conclusion")
    if winner:
        pw_w = next(p for p in mv_results["pairwise"] if p["name"] == winner)
        if pw_w["uplift_cr"] > 0 and pw_w["uplift_rpv"] > 0:
            report.append(f"🟢 **{winner} — Growth Engine:** CR and RPV both up. Ship.")
        elif pw_w["uplift_cr"] > 0 and pw_w["uplift_rpv"] < 0:
            report.append(f"🟡 **{winner} — Volume Play:** CR up, RPV down. Check revenue before shipping.")
        else:
            report.append(f"🟡 **{winner} — Marginal:** Review full metrics before shipping.")
    else:
        report.append("🔴 No variant qualifies for shipping on current data.")

    return "\n\n".join(report)


# -----------------------------------------------
# PLOTTING
# -----------------------------------------------
def plot_multivariant_bar(title, values, labels, unit=""):
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.2), 5))
    ctrl_val = values[0]
    colors   = [
        GROUP_COLORS[i] if (i == 0 or v >= ctrl_val) else "#d62728"
        for i, v in enumerate(values)
    ]
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    mx   = max(values) if max(values) > 0 else 1
    ax.set_ylim(0, mx * 1.18)
    ax.set_title(f"{title} by Group")
    ax.set_ylabel(title)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + mx * 0.01,
                f"{unit}{v:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    st.pyplot(fig)
    plt.close(fig)

def plot_strategic_matrix(metrics_list):
    fig, ax = plt.subplots(figsize=(9, 6))
    ctrl = metrics_list[0]
    for i, m in enumerate(metrics_list):
        ax.scatter(m["cr_pct"], m["aov"], color=GROUP_COLORS[i], s=220,
                   label=m["name"], zorder=5)
        if i > 0:
            ax.annotate("",
                xy=(m["cr_pct"], m["aov"]), xytext=(ctrl["cr_pct"], ctrl["aov"]),
                arrowprops=dict(arrowstyle="->", color=GROUP_COLORS[i], lw=1.5, ls="--"),
            )
    ax.axvline(ctrl["cr_pct"], color="gray", ls=":", alpha=0.4)
    ax.axhline(ctrl["aov"],    color="gray", ls=":", alpha=0.4)
    xl, yl = ax.get_xlim(), ax.get_ylim()
    cx, cy = ctrl["cr_pct"], ctrl["aov"]
    for label, xpos, ypos, colour in [
        ("Growth\nCR↑ AOV↑",  cx+(xl[1]-cx)*.5, cy+(yl[1]-cy)*.5, "green"),
        ("Premium\nCR↓ AOV↑", cx-(cx-xl[0])*.5, cy+(yl[1]-cy)*.5, "orange"),
        ("Volume\nCR↑ AOV↓",  cx+(xl[1]-cx)*.5, cy-(cy-yl[0])*.5, "orange"),
        ("Loss\nCR↓ AOV↓",    cx-(cx-xl[0])*.5, cy-(cy-yl[0])*.5, "red"),
    ]:
        ax.text(xpos, ypos, label, ha="center", va="center",
                fontsize=8, color=colour, alpha=0.5)
    ax.set_title("Strategic Matrix: CR vs AOV")
    ax.set_xlabel("Conversion Rate (%)")
    ax.set_ylabel("Average Order Value ($)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

def plot_bayesian_pdfs(groups):
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, g in enumerate(groups):
        a = g["conv"] + 1
        b = max(g["users"] - g["conv"], 0) + 1
        d = beta(a, b)
        x = np.linspace(d.ppf(0.001), d.ppf(0.999), 1000)
        ax.plot(x, d.pdf(x), label=g["name"], color=GROUP_COLORS[i], lw=2)
        ax.fill_between(x, d.pdf(x), 0, alpha=0.15, color=GROUP_COLORS[i])
    ax.set_title("Bayesian Posterior Distributions")
    ax.set_xlabel("Conversion Rate")
    ax.set_ylabel("Probability Density")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

def run_bootstrap_and_plot(uc, cc, uv, cv, alpha_val=0.05, label_v="Variation"):
    rng   = np.random.default_rng()
    pc    = safe_divide(cc, uc)
    pv    = safe_divide(cv, uv)
    sim_c = rng.binomial(uc, pc, BOOTSTRAP_SAMPLES) / uc
    sim_v = rng.binomial(uv, pv, BOOTSTRAP_SAMPLES) / uv
    diffs = sim_v - sim_c
    ci_l  = float(np.percentile(diffs, alpha_val / 2 * 100))
    ci_h  = float(np.percentile(diffs, (1 - alpha_val / 2) * 100))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(diffs, bins=60, color="orange", edgecolor="black", alpha=0.7)
    ax.axvline(ci_l, color="red", ls="--", label=f"{(1-alpha_val)*100:.0f}% CI Lower")
    ax.axvline(ci_h, color="red", ls="--", label=f"{(1-alpha_val)*100:.0f}% CI Upper")
    ax.axvline(0,    color="black", lw=1.2, label="No effect")
    ax.set_title(f"Bootstrap CR Difference ({label_v} − Control)")
    ax.set_xlabel("Difference in Conversion Rate")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    return sim_c * 100, sim_v * 100, ci_l * 100, ci_h * 100

def plot_box_plots(samples_c, samples_v, label_v="Best Variation"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([samples_c, samples_v], patch_artist=True, widths=0.6,
               boxprops=dict(facecolor="skyblue"))
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Control", label_v])
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_title("Box Plot: Bootstrap CR Distributions")
    st.pyplot(fig)
    plt.close(fig)

def plot_power_curve(base_cr, mdes, alpha, max_n):
    fig, ax = plt.subplots(figsize=(10, 6))
    sample_sizes = np.linspace(100, max_n, 100)
    
    for i, mde in enumerate(mdes):
        relative_effect = mde / 100.0
        p1 = base_cr / 100.0
        p2 = p1 * (1 + relative_effect)
        es = proportion_effectsize(p1, p2)
        
        powers = []
        for n in sample_sizes:
            # solve_power returns power for a given sample size
            power = NormalIndPower().solve_power(effect_size=es, nobs1=n, alpha=alpha, ratio=1)
            powers.append(power)
            
        color = GROUP_COLORS[i % len(GROUP_COLORS)]
        ax.plot(sample_sizes, powers, label=f"MDE: {mde}%", lw=2, color=color)

    ax.axhline(0.80, color="red", ls="--", label="80% Power Threshold")
    ax.set_title(f"Statistical Power over Sample Size (Base CR: {base_cr}%)")
    ax.set_xlabel("Sample Size (per group)")
    ax.set_ylabel("Power")
    ax.set_ylim(0, 1.05)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown(
    f'<div style="display:flex;align-items:center;">{ICON_SETTINGS}'
    '<h2 style="display:inline;font-size:1.5rem;margin-left:5px;">Settings</h2></div>',
    unsafe_allow_html=True,
)

with st.sidebar.expander("Save & Load Analysis", expanded=False):
    st.info("Snapshot your experiment. Download Day 7, re-upload Day 14 to track evolution.")
    snapshot = {k: st.session_state.get(k) for k in SAVE_KEYS}
    st.download_button("Download Inputs (.json)", json.dumps(snapshot, indent=2),
                       "experiment_snapshot.json", "application/json")
    uploaded = st.file_uploader("Load Snapshot", type=["json"])
    if uploaded is not None:
        try:
            loaded = json.load(uploaded)
            for k in SAVE_KEYS:
                if k in loaded:
                    st.session_state[k] = loaded[k]
            st.success("Snapshot loaded!")
            st.rerun()
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            st.error(f"Could not load file: {e}")

confidence_level = st.sidebar.selectbox(
    "Confidence Level", ["95%", "90%", "99%"], index=0, key="conf_level",
    help="⚡ 90% — Fast decisions  |  ⚖️ 95% — Standard  |  🐢 99% — High-stakes",
)
alpha = {"90%": 0.10, "95%": 0.05, "99%": 0.01}[confidence_level]

st.sidebar.markdown("---")

# --- Experiment Planning ---
st.sidebar.markdown(
    f'<div style="display:flex;align-items:center;">{ICON_CALENDAR}'
    '<h2 style="display:inline;font-size:1.5rem;margin-left:5px;">Experiment Planning</h2></div>',
    unsafe_allow_html=True,
)
with st.sidebar.expander("Sample Size Calculator", expanded=False):
    st.caption(f"Plan required traffic at {confidence_level} confidence, 80% power.")
    plan_traffic  = st.number_input("Traffic (Last 28 Days)", step=1000,  key="p_traffic")
    plan_base_cr  = st.number_input("Baseline CR (%)",        step=0.1,   key="p_base_cr")
    plan_base_aov = st.number_input("Baseline AOV ($)",       step=1.0,   key="p_base_aov")
    plan_mde      = st.number_input("Min Detectable Effect (%)", step=0.5, key="p_mde")
    volatility    = st.selectbox("Revenue Variance",
        ["Low (Subscription)", "Medium (Standard E-com)", "High (Whales/B2B)"],
        index=1, key="p_vol")
    sd_mult = {"Low": 1.0, "Medium": 2.0, "High": 3.0}[volatility.split()[0]]
    if st.button("Calculate Duration"):
        if plan_base_cr <= 0 or plan_base_aov <= 0 or plan_traffic <= 0:
            st.error("Traffic, Baseline CR, and AOV must all be > 0.")
        else:
            daily   = plan_traffic / 28
            p1      = plan_base_cr / 100
            p2      = p1 * (1 + plan_mde / 100)
            es_cr   = proportion_effectsize(p1, p2)
            n_cr    = NormalIndPower().solve_power(effect_size=es_cr, alpha=alpha, power=0.8, ratio=1)
            est_sd  = plan_base_aov * sd_mult
            es_rpv  = safe_divide(plan_base_aov * plan_mde / 100, est_sd)
            if es_rpv > 0:
                n_rpv     = TTestIndPower().solve_power(effect_size=es_rpv, alpha=alpha, power=0.8, ratio=1)
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

# --- Data Inputs ---
st.sidebar.markdown(
    f'<div style="display:flex;align-items:center;">{ICON_TROPHY}'
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
days_run = st.sidebar.number_input("Days Test Ran", min_value=1, key="days")

if num_variations > 1:
    st.sidebar.markdown("---")
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


# ============================================================
# BUILD GROUPS & RUN ANALYSIS
# ============================================================
groups = [{"name": "Control", "users": users_c, "conv": conv_c, "rev": rev_c, "prod": prod_c}] + var_inputs

mv      = run_multivariate_analysis(groups, alpha, mc_method)
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

# ---- Pre-compute bootstrap samples at top level so both Tab 9 & Tab 10
#      can share the same data without cross-tab session-state coupling. ----
_boot_rng   = np.random.default_rng()
_boot_pc    = safe_divide(ctrl_m["conv"], ctrl_m["users"])
_boot_pv    = safe_divide(best_m["conv"], best_m["users"])
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
        "p (adj)":     f"{pw['p_adjusted']:.4f}" if pw else "—",
        "Significant": ("✅" if pw["significant"] else "❌") if pw else "—",
    })
st.dataframe(pd.DataFrame(kpi_rows), use_container_width=True, hide_index=True)

# Winner callout
if mv["winner"]:
    st.success(f"🏆 **Winner: {mv['winner']}** — highest significant CR uplift after {mc_method.upper()} correction.")
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
with st.expander(f"📊 Revenue Significance — Control vs {best_m['name']}", expanded=False):
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
hc1, hc2 = st.columns(2)
with hc1:
    if p_srm < 0.01:
        st.error(f"⚠️ SRM DETECTED (p={p_srm:.4f}) — traffic split uneven. Results may be invalid.")
    else:
        st.success(f"✅ SRM PASSED (p={p_srm:.4f})")
with hc2:
    if days_run < 7:
        st.error(f"⚠️ Too short ({days_run} days) — results unreliable.")
    elif days_run < 14:
        st.warning(f"⚠️ Short duration ({days_run} days) — watch for novelty effects.")
    else:
        st.success(f"✅ Duration OK ({days_run} days)")

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "Smart Analysis", "AI Analysis", "Stopping & Sequential",
    "Strategic Matrix", "Product Metrics", "Revenue Charts",
    "CR Comparison", "Bayesian", "Bootstrap", "Box Plot", "Power Curves"
])

# ---- TAB 1: SMART ANALYSIS ----
with tab1:
    st.markdown("### Auto-Analyst Report")
    st.info("Instant rule-based analysis — no API key required.")
    hyp_smart = st.text_area("Hypothesis:", placeholder="We believed that...", height=70, key="hyp_smart")
    if st.button("Generate Smart Report"):
        smart_payload = {
            "days": days_run,
            "p_srm": p_srm,
        }
        st.markdown("---")
        st.markdown(generate_smart_analysis(hyp_smart, mv, bayes, smart_payload, alpha))

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
            )
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
