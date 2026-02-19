import streamlit as st
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import beta, chisquare
from statsmodels.stats.proportion import proportions_ztest, proportion_confint, proportion_effectsize
from statsmodels.stats.power import NormalIndPower, TTestIndPower
import openai

# -----------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------
st.set_page_config(
    page_title="A/B Test Analyzer v1.8.0c",
    layout="wide",
    initial_sidebar_state="expanded"
)
matplotlib.use("Agg")  # Prevents threading crashes in Streamlit

# -----------------------------------------------
# SVG ICONS
# -----------------------------------------------
ICON_SETTINGS = """<svg viewBox="0 0 1024 1024" style="width:1.5em;height:1.5em;vertical-align:middle;margin-right:10px;" xmlns="http://www.w3.org/2000/svg"><path d="M512 512m-10 0a10 10 0 1 0 20 0 10 10 0 1 0-20 0Z" fill="#E73B37"/><path d="M512 306.8c27.7 0 54.6 5.4 79.8 16.1 24.4 10.3 46.4 25.1 65.2 44s33.6 40.8 44 65.2c10.7 25.3 16.1 52.1 16.1 79.8 0 27.7-5.4 54.6-16.1 79.8-10.3 24.4-25.1 46.4-44 65.2-18.8 18.8-40.8 33.6-65.2 44-25.3 10.7-52.1 16.1-79.8 16.1-27.7 0-54.6-5.4-79.8-16.1-24.4-10.3-46.4-25.1-65.2-44-18.8-18.8-33.6-40.8-44-65.2-10.7-25.3-16.1-52.1-16.1-79.8 0-27.7 5.4-54.6 16.1-79.8 10.3-24.4 25.1-46.4 44-65.2s40.8-33.6 65.2-44c25.2-10.6 52.1-16.1 79.8-16.1m0-22c-125.4 0-227.1 101.7-227.1 227.1S386.6 739.1 512 739.1c125.4 0 227.1-101.7 227.1-227.1S637.4 284.8 512 284.8z" fill="#ffffff"/><path d="M544.2 107.3l34.1 92.3 7.4 19.9 20.2 6.6c10.3 3.4 32.1 12.9 43.4 18.1l18.7 8.6 18.6-8.9 87.9-41.8 46.4 46.5-41.2 89.4-8.9 19.3 9.6 19c6.8 13.4 12.6 27.5 17.4 41.9l6.7 20.5 20.3 7.2 91.7 32.6v65.7l-92.3 34.1-19.9 7.4-6.6 20.2c-4.7 14.4-10.6 28.4-17.4 41.9l-9.8 19.3 9.3 19.5 41.8 87.9-46.5 46.5-89.1-41.3-19.3-8.9-19 9.6c-13.4 6.8-27.5 12.6-41.9 17.4l-20.5 6.7-7.2 20.3-32.6 91.7h-65.7l-34.1-92.3-7.4-19.9-20.2-6.6c-10.3-3.4-32.1-12.9-43.4-18.1L356 771l-18.6 8.9-87.9 41.8-46.4-46.5 41.2-89.3 8.9-19.3-9.6-19c-6.8-13.4-12.6-27.5-17.4-41.9l-6.7-20.5-20.3-7.2-91.7-32.6v-65.7l92.3-34.1 19.9-7.4 6.6-20.2c3.4-10.3 12.9-32.1 18.1-43.4l8.6-18.7-8.9-18.6-41.8-87.9 46.4-46.4 89.3 41.2 19.3 8.9 19-9.6c13.4-6.8 27.5-12.6 41.9-17.4l20.5-6.7 7.2-20.3 32.6-91.7h65.7m30.7-44.1H447.4l-43 121c-16.6 5.5-32.7 12.1-48.1 19.9l-117.2-54-90.1 90.1 55.2 116s-14.5 31.4-19.9 48.1l-121 44.7v127.4l121 43c5.5 16.6 12.1 32.6 19.9 48l-54 117.2 90.1 90.1 116-55.2s31.4 14.5 48.1 19.9l44.7 121h127.4l43-121c16.6-5.5 32.6-12.1 48-19.9l117.2 54 90.1-90.1-55.2-116c7.8-15.4 14.5-31.4 19.9-48l121-44.7V447.4l-121-43c-5.5-16.6-12.1-32.6-19.9-48l54-117.2-90.1-90.1-115.9 55.2s-31.5-14.5-48.1-19.9L574.9 63.3z" fill="#ffffff"/></svg>"""

ICON_CALENDAR = """<svg viewBox="0 0 1024 1024" style="width:1.5em;height:1.5em;vertical-align:middle;margin-right:10px;" xmlns="http://www.w3.org/2000/svg"><path d="M716 190.9v-67.8h-44v67.8H352v-67.8h-44v67.8H92v710h840v-710H716z m-580 44h172v69.2h44v-69.2h320v69.2h44v-69.2h172v151.3H136V234.9z m752 622H136V402.2h752v454.7z" fill="#ffffff"/><path d="M319 565.7m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"/><path d="M510 565.7m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"/><path d="M701.1 565.7m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"/><path d="M319 693.4m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"/><path d="M510 693.4m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"/><path d="M701.1 693.4m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"/></svg>"""

ICON_BAR_CHART = """<svg viewBox="0 0 1024 1024" style="width:1.5em;height:1.5em;vertical-align:middle;margin-right:10px;" xmlns="http://www.w3.org/2000/svg"><path d="M928.1 881v44H95.9V99h44v782z" fill="#ffffff"/><path d="M352 435.7v403.4H204V435.7h148m22-22H182v447.4h192V413.7zM608 307.9v531.2H460V307.9h148m22-22H438v575.2h192V285.9z" fill="#ffffff"/><path d="M866.1 177.3v663.9H714V177.3h152.1m20-20H694v703.9h192V157.3h0.1z" fill="#E73B37"/></svg>"""

ICON_PIE = """<svg viewBox="0 0 1024 1024" style="width:1.5em;height:1.5em;vertical-align:middle;margin-right:10px;" xmlns="http://www.w3.org/2000/svg"><path d="M429.9 186.7v406.4h407.5c-4 34.1-12.8 67.3-26.2 99.1-18.4 43.6-44.8 82.7-78.5 116.3-33.6 33.6-72.8 60-116.4 78.4-45.1 19.1-93 28.7-142.5 28.7-49.4 0-97.4-9.7-142.5-28.7-43.6-18.4-82.7-44.8-116.4-78.4-33.6-33.6-60-72.7-78.4-116.3-19.1-45.1-28.7-93-28.7-142.4s9.7-97.3 28.7-142.4c18.4-43.6 44.8-82.7 78.4-116.3 33.6-33.6 72.8-60 116.4-78.4 31.7-13.2 64.7-21.9 98.6-26m44-46.6c-226.4 0-410 183.5-410 409.8s183.6 409.8 410 409.8 410-183.5 410-409.8v-0.8h-410v-409z" fill="#ffffff"/><path d="M566.1 80.5c43.7 1.7 86.4 10.6 127 26.4 44 17.1 84.2 41.8 119.6 73.5 71.7 64.1 117.4 151.7 128.7 246.7 1.2 9.9 2 20 2.4 30.2H566.1V80.5m-16-16.3v409h410c0-16.3-1-32.3-2.9-48.1C933.1 221.9 760 64.2 550.1 64.2zM264.7 770.4c-23.1-23.1-42.3-49.1-57.3-77.7l-14.7 6.5c35.7 68.2 94 122.7 165 153.5l4.3-15.6c-36.3-16-69.1-38.4-97.3-66.7z" fill="#E73B37"/></svg>"""

ICON_BRAIN = """<svg viewBox="0 0 1024 1024" style="width:1.5em;height:1.5em;vertical-align:middle;margin-right:10px;" xmlns="http://www.w3.org/2000/svg"><path d="M512 301.2m-10 0a10 10 0 1 0 20 0 10 10 0 1 0-20 0Z" fill="#E73B37"/><path d="M511.8 256.6c24.4 0 44.2 19.8 44.2 44.2S536.2 345 511.8 345s-44.2-19.8-44.2-44.2 19.9-44.2 44.2-44.2m0-20c-35.5 0-64.2 28.7-64.2 64.2s28.7 64.2 64.2 64.2 64.2-28.7 64.2-64.2-28.7-64.2-64.2-64.2z" fill="#E73B37"/><path d="M730.7 529.5c0.4-8.7 0.6-17.4 0.6-26.2 0-179.6-86.1-339.1-219.3-439.5-133.1 100.4-219.2 259.9-219.2 439.5 0 8.8 0.2 17.5 0.6 26.1-56 56-90.6 133.3-90.6 218.7 0 61.7 18 119.1 49.1 167.3 30.3-49.8 74.7-90.1 127.7-115.3 39-18.6 82.7-29 128.8-29 48.3 0 93.9 11.4 134.3 31.7 52.5 26.3 96.3 67.7 125.6 118.4 33.4-49.4 52.9-108.9 52.9-173.1 0-85.4-34.6-162.6-90.5-218.6z" fill="#ffffff"/><path d="M512 819.3c8.7 0 24.7 22.9 24.7 60.4s-16 60.4-24.7 60.4-24.7-22.9-24.7-60.4 16-60.4 24.7-60.4m0-20c-24.7 0-44.7 36-44.7 80.4 0 44.4 20 80.4 44.7 80.4s44.7-36 44.7-80.4c0-44.4-20-80.4-44.7-80.4z" fill="#E73B37"/></svg>"""

ICON_UPLOAD = """<svg viewBox="0 0 1024 1024" style="width:1.5em;height:1.5em;vertical-align:middle;margin-right:10px;" xmlns="http://www.w3.org/2000/svg"><path d="M220.5 245.4c-32.8 32.8-55.1 73.2-65.2 117.3h16.5c18.8-75.3 75.1-135.9 148-160.7v-16.9c-37.1 11.6-71 32-99.3 60.3z" fill="#E73B37"/><path d="M959.9 540.8c0 113.6-92.1 205.8-205.7 205.9H590.9v-44h163.3c43.2 0 83.8-16.9 114.3-47.4 30.6-30.6 47.4-71.2 47.4-114.5 0-43.2-16.8-83.9-47.4-114.4S797.2 379 754 379c-11.5 0-22.8 1.2-33.8 3.5-15 3.2-29.4 8.4-42.8 15.7-1-15.4-3.3-30.7-6.8-45.6-3.6-15.6-8.6-30.8-14.9-45.7-14.4-33.9-34.9-64.4-61.1-90.6-26.2-26.2-56.6-46.7-90.6-61.1-35.1-14.8-72.4-22.4-110.9-22.4s-75.8 7.5-110.9 22.4c-33.9 14.3-64.4 34.9-90.6 61.1-26.2 26.2-46.7 56.7-61.1 90.6-14.9 35.1-22.4 72.4-22.4 110.9s7.5 75.8 22.4 110.9c14.3 33.9 34.9 64.4 61.1 90.6 26.2 26.2 56.7 46.7 90.6 61.1 35.1 14.8 72.4 22.4 110.9 22.4h39.7v44h-41C210.7 746 64.1 599 64.1 417.7c0-181.7 147.3-329 329-329 154.6 0 284.3 106.6 319.5 250.3 13.4-2.7 27.2-4.2 41.4-4.2 113.7 0.1 205.9 92.2 205.9 205.9z" fill="#ffffff"/><path d="M692.9 636.1h-22.6L519.8 485.6v449.6h-16V485.8L353.4 636.1h-22.6l181-181z" fill="#E73B37"/></svg>"""

ICON_TROPHY = """<svg viewBox="0 0 1024 1024" style="width:1.5em;height:1.5em;vertical-align:middle;margin-right:10px;" xmlns="http://www.w3.org/2000/svg"><path d="M828.5 180.1h-9.9v-54.7h23.5v-44H182v44h23v54.7h-9.5C123.2 180.1 64 239.2 64 311.5v0.1c0 72.3 59.2 131.5 131.5 131.5h9.6c0 1.3 0.1 2.5 0.1 3.7 0.5 17.7 2.7 35.4 6.2 52.5 17.8 85.7 71.8 160 148.3 204 4.8 2.8 9.8 5.4 14.7 7.9 15.3 7.7 31.2 14.1 47.4 19.2 3.4 1 6.8 2 10.2 2.9v165.2H250.4v44h511.9v-44H591.9V733.4c3.7-1 7.3-2.1 10.9-3.2 16.2-5.1 32.2-11.6 47.4-19.4 5-2.5 10-5.3 14.8-8.1 75.6-43.9 129.2-117.8 147-202.7 3.6-17.2 5.8-34.9 6.3-52.4 0.1-1.5 0.1-3 0.1-4.5h10c72.3 0 131.5-59.2 131.5-131.5v-0.1c0.1-72.3-59.1-131.4-131.4-131.4zM205 399.2h-9.5c-23.2 0-45.1-9.1-61.7-25.7s-25.7-38.5-25.7-61.7v-0.1c0-23.2 9.1-45.2 25.7-61.7 16.6-16.6 38.5-25.7 61.7-25.7h9.5v174.9z m370.9 499.4h-128V737.3c20.9 4.5 42.3 6.8 63.9 6.8 21.7 0 43.1-2.3 64.1-6.8v161.3z m198.7-461.4c0 2.9 0 5.9-0.2 8.9-0.5 15-2.3 30.1-5.4 44.9-15.3 72.7-61.2 136-126.1 173.7-4.1 2.4-8.4 4.7-12.7 6.9-13 6.6-26.7 12.2-40.6 16.6-25.2 7.9-51.4 11.9-77.9 11.9-26.2 0-52.2-3.9-77.1-11.6-13.9-4.3-27.5-9.8-40.6-16.4-4.2-2.1-8.5-4.4-12.6-6.8-65.4-37.8-111.7-101.5-126.9-174.8-3.1-14.7-4.9-29.8-5.3-45-0.1-2.7-0.1-5.5-0.1-8.2v-312h525.6v311.9zM916 311.7c0 23.2-9.1 45.2-25.7 61.7-16.6 16.6-38.5 25.7-61.7 25.7h-9.9v-175h9.9c23.2 0 45.1 9.1 61.7 25.7s25.7 38.5 25.7 61.7v0.2z" fill="#ffffff"/><path d="M555.4 659.6l-4.8-19.4c0.3-0.1 26.5-6.8 55.4-23.5 37.8-21.9 62-49.7 72-82.7l19.1 5.8c-11.4 37.6-39.6 70.3-81.6 94.5-31.2 18-58.9 25-60.1 25.3z" fill="#E73B37"/></svg>"""

# -----------------------------------------------
# CONSTANTS
# -----------------------------------------------
# Number of Monte Carlo samples for Bayesian analysis.
# Higher = more accurate but slower. 50,000 is a good balance.
BAYESIAN_SAMPLES = 50_000

# Number of bootstrap resamples for the CI histogram.
BOOTSTRAP_SAMPLES = 10_000


# -----------------------------------------------
# STATE INITIALIZATION
# -----------------------------------------------
def initialize_state():
    defaults = {
        "users_c": 5000, "conv_c": 500, "rev_c": 25000.0, "prod_c": 750,
        "users_v": 5000, "conv_v": 600, "rev_v": 33000.0, "prod_v": 1000,
        "days": 14,
        "conf_level": "95%",
        "p_traffic": 50000, "p_base_cr": 2.5, "p_base_aov": 75.0, "p_mde": 5.0,
        "p_vol": "Medium (Standard E-com)",
        "s1_uc": 2000, "s1_cc": 100, "s1_uv": 2000, "s1_cv": 110,
        "s2_uc": 3000, "s2_cc": 400, "s2_uv": 3000, "s2_cv": 420,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


# -----------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------
def render_header(svg_code: str, text: str, level: int = 2):
    st.markdown(
        f"""<div style="display:flex;align-items:center;margin-bottom:10px;">
        {svg_code}<h{level} style="margin:0;padding:0;">{text}</h{level}>
        </div>""",
        unsafe_allow_html=True,
    )


def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Division that returns `fallback` instead of raising ZeroDivisionError."""
    return numerator / denominator if denominator != 0 else fallback


def calculate_uplift(ctrl: float, var: float) -> float:
    """Percentage uplift of var over ctrl. Returns 0 if ctrl is 0."""
    return safe_divide((var - ctrl) * 100, ctrl)


# FIX: Use scipy.stats.chisquare (goodness-of-fit) instead of
# chi2_contingency (independence test), which was the wrong function for SRM.
def perform_srm_test(observed: list, expected_split: tuple = (0.5, 0.5)):
    """
    Sample Ratio Mismatch test.
    Returns (chi2_statistic, p_value).
    p < 0.01 indicates a likely traffic allocation problem.
    """
    total = sum(observed)
    expected = [total * p for p in expected_split]
    stat, p_value = chisquare(observed, f_exp=expected)
    return stat, p_value


def calculate_bayesian_risk(alpha_c, beta_c, alpha_v, beta_v):
    """
    Monte Carlo Bayesian analysis using Beta distributions.
    Returns (prob_variation_wins, expected_loss_switching, expected_loss_staying).
    """
    rng = np.random.default_rng()  # Use a modern, seedable RNG
    s_c = rng.beta(alpha_c, beta_c, BAYESIAN_SAMPLES)
    s_v = rng.beta(alpha_v, beta_v, BAYESIAN_SAMPLES)
    prob_v_wins = float(np.mean(s_v > s_c))
    loss_v = float(np.mean(np.maximum(s_c - s_v, 0)))
    loss_c = float(np.mean(np.maximum(s_v - s_c, 0)))
    return prob_v_wins, loss_v, loss_c


def check_simpsons_paradox(seg1: dict, seg2: dict):
    """
    Detects Simpson's Paradox across two segments.
    Returns (paradox_detected, uplift_seg1, uplift_seg2, uplift_aggregate).
    """
    cr_c1 = safe_divide(seg1["conv_c"], seg1["users_c"])
    cr_v1 = safe_divide(seg1["conv_v"], seg1["users_v"])
    up1 = calculate_uplift(cr_c1, cr_v1)

    cr_c2 = safe_divide(seg2["conv_c"], seg2["users_c"])
    cr_v2 = safe_divide(seg2["conv_v"], seg2["users_v"])
    up2 = calculate_uplift(cr_c2, cr_v2)

    agg_uc = seg1["users_c"] + seg2["users_c"]
    agg_cc = seg1["conv_c"] + seg2["conv_c"]
    agg_uv = seg1["users_v"] + seg2["users_v"]
    agg_cv = seg1["conv_v"] + seg2["conv_v"]

    cr_c_agg = safe_divide(agg_cc, agg_uc)
    cr_v_agg = safe_divide(agg_cv, agg_uv)
    up_agg = calculate_uplift(cr_c_agg, cr_v_agg)

    paradox = (up1 > 0 and up2 > 0 and up_agg < 0) or (up1 < 0 and up2 < 0 and up_agg > 0)
    return paradox, up1, up2, up_agg


# -----------------------------------------------
# AI ANALYSIS
# -----------------------------------------------
def get_ai_analysis(api_key: str, hypothesis: str, metrics: dict, provider: str = "OpenAI", conf_level: str = "95%") -> str:
    if not api_key:
        return "Please enter a valid API Key to generate this analysis."

    base_url = "https://api.deepseek.com" if provider == "DeepSeek" else "https://api.openai.com/v1"
    model_name = "deepseek-reasoner" if provider == "DeepSeek" else "gpt-4o"

    prompt = f"""Analyze this A/B test result.
Configuration: Confidence Level: {conf_level}
Hypothesis: "{hypothesis}"
Data: {metrics}
Task: Provide an Executive Summary, Trade-off Analysis, Risk Assessment, and Recommendation.
Format your response in Markdown."""

    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"**Error connecting to AI provider:** {e}"


# -----------------------------------------------
# RULE-BASED SMART ANALYSIS
# -----------------------------------------------
def generate_smart_analysis(hypothesis: str, metrics: dict, alpha_val: float) -> str:
    report = []

    # --- Guard: SRM detected means results are invalid ---
    if metrics["p_srm"] < 0.01:
        report.append("### ⛔ CRITICAL: Sample Ratio Mismatch (SRM) Detected")
        report.append(
            f"SRM p-value = **{metrics['p_srm']:.4f}** (threshold: 0.01). "
            "The traffic split between Control and Variation is not what was intended. "
            "**All results below are likely invalid.** Investigate your randomisation logic before drawing any conclusions."
        )
        return "\n\n".join(report)

    # --- Executive Summary ---
    sig = metrics["p_cr"] <= alpha_val
    uplift = metrics["uplift_cr"]

    if sig and uplift > 0:
        headline = "✅ WINNER: Statistically Significant Positive Result"
        summary = f"The Variation outperforms Control with **{(1 - alpha_val)*100:.0f}% confidence** (p = {metrics['p_cr']:.4f})."
    elif sig and uplift < 0:
        headline = "❌ LOSER: Statistically Significant Negative Result"
        summary = f"The Variation is performing **significantly worse** than Control (p = {metrics['p_cr']:.4f}). Do not ship."
    else:
        headline = "⚠️ INCONCLUSIVE: No Clear Winner"
        summary = f"We cannot reject the null hypothesis (p = {metrics['p_cr']:.4f} > {alpha_val}). More data is needed."

    report.append(f"### {headline}")
    report.append(summary)

    if hypothesis:
        report.append(f"**Hypothesis tested:** _{hypothesis}_")

    # --- Data Health ---
    report.append("### Data Health & Validity")
    days = metrics["days"]
    if days < 7:
        report.append(f"- ⚠️ **Duration Warning:** Only {days} days — too short, results are unreliable.")
    elif days < 14:
        report.append(f"- ⚠️ **Duration Caution:** {days} days — watch for novelty effects.")
    else:
        report.append(f"- ✅ **Duration:** {days} days — healthy.")
    report.append(f"- ✅ **SRM Check:** Passed (p = {metrics['p_srm']:.4f}).")

    # --- Performance Breakdown ---
    report.append("### Performance Breakdown")
    cr_dir = "Improved" if uplift > 0 else "Decreased"
    report.append(f"- **Conversion Rate:** {cr_dir} by **{uplift:.2f}%** ({metrics['cr_c']:.2f}% → {metrics['cr_v']:.2f}%).")

    aov_change = metrics["uplift_aov"]
    if abs(aov_change) < 1.0:
        aov_text = "remained stable"
    elif aov_change > 0:
        aov_text = f"increased by {aov_change:.2f}%"
    else:
        aov_text = f"decreased by {abs(aov_change):.2f}%"
    report.append(f"- **Average Order Value:** {aov_text} (${metrics['aov_c']:.2f} → ${metrics['aov_v']:.2f}).")

    rpv_delta = metrics["rpv_v"] - metrics["rpv_c"]
    rpv_dir = "more" if rpv_delta >= 0 else "less"
    report.append(f"- **Revenue Per Visitor:** ${abs(rpv_delta):.2f} {rpv_dir} per visitor.")

    # --- Bayesian Risk ---
    report.append("### Bayesian Risk Assessment")
    report.append(f"- **Probability Variation is Best:** {metrics['prob_v_wins']:.1f}%")
    report.append(f"- **Expected Loss (if you switch):** {metrics['loss_v']:.5f}%")
    report.append(f"- **Expected Loss (if you stay):** {metrics['loss_c']:.5f}%")

    # --- Confidence Interval ---
    report.append("### Confidence Interval on Conversion Rate Difference")
    ci_low, ci_high = metrics["ci_low"], metrics["ci_high"]
    if ci_low > 0:
        report.append(f"The {(1-alpha_val)*100:.0f}% CI ({ci_low:.2f}% to {ci_high:.2f}%) is **entirely positive** — strong evidence the Variation wins.")
    elif ci_high < 0:
        report.append(f"The {(1-alpha_val)*100:.0f}% CI ({ci_low:.2f}% to {ci_high:.2f}%) is **entirely negative** — strong evidence the Variation loses.")
    else:
        report.append(f"The {(1-alpha_val)*100:.0f}% CI ({ci_low:.2f}% to {ci_high:.2f}%) **crosses zero** — uncertainty remains.")

    # --- Strategic Quadrant ---
    report.append("### Strategic Conclusion")
    if uplift > 0 and aov_change > 0:
        report.append("🟢 **Growth Engine:** Both conversion volume and basket value are up. This is a clean win.")
    elif uplift > 0 and aov_change < 0:
        report.append("🟡 **Discount Effect:** More orders but smaller baskets. Check if RPV is still positive before shipping.")
    elif uplift < 0 and aov_change > 0:
        report.append("🟡 **Premium Shift:** Fewer but higher-value orders. May suit a premium positioning strategy.")
    else:
        report.append("🔴 **Negative Friction:** Both conversion and order value are down. Do not ship.")

    return "\n\n".join(report)


# -----------------------------------------------
# PLOTTING FUNCTIONS
# -----------------------------------------------
def plot_strategic_matrix(cr_c, aov_c, cr_v, aov_v):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cr_c, aov_c, color="blue", s=200, label="Control", zorder=5)
    ax.scatter(cr_v, aov_v, color="green", s=200, label="Variation", zorder=5)
    ax.annotate(
        "", xy=(cr_v, aov_v), xytext=(cr_c, aov_c),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, ls="--"),
    )
    ax.axvline(cr_c, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(aov_c, color="gray", linestyle=":", alpha=0.5)
    ax.set_title("Strategic Matrix: Volume (CR) vs Value (AOV)")
    ax.set_xlabel("Conversion Rate (%)")
    ax.set_ylabel("Average Order Value ($)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)  # FIX: Prevent memory leak


def plot_metric_comparison(name: str, val_c: float, val_v: float, unit: str = ""):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4", "#2ca02c"] if val_v >= val_c else ["#1f77b4", "#d62728"]
    ax.bar(["Control", "Variation"], [val_c, val_v], color=colors, alpha=0.8)
    ax.set_title(f"{name} Comparison")
    ax.set_ylabel(name)
    max_val = max(val_c, val_v)
    ax.set_ylim(0, max_val * 1.15 if max_val > 0 else 1)
    for i, v in enumerate([val_c, val_v]):
        ax.text(i, v + (max_val * 0.01 if max_val > 0 else 0.01), f"{unit}{v:.2f}",
                ha="center", va="bottom", fontweight="bold")
    st.pyplot(fig)
    plt.close(fig)  # FIX: Prevent memory leak


def plot_bayesian_pdfs(alpha_c, beta_c, alpha_v, beta_v):
    dist_c = beta(alpha_c, beta_c)
    dist_v = beta(alpha_v, beta_v)
    x_min = min(dist_c.ppf(0.001), dist_v.ppf(0.001))
    x_max = max(dist_c.ppf(0.999), dist_v.ppf(0.999))
    x = np.linspace(x_min, x_max, 1000)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, dist_c.pdf(x), label="Control", color="blue")
    ax.fill_between(x, dist_c.pdf(x), 0, alpha=0.3, color="blue")
    ax.plot(x, dist_v.pdf(x), label="Variation", color="green")
    ax.fill_between(x, dist_v.pdf(x), 0, alpha=0.3, color="green")
    ax.set_title("Bayesian Posterior Distributions")
    ax.set_xlabel("Conversion Rate")
    ax.set_ylabel("Probability Density")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)  # FIX: Prevent memory leak


def run_bootstrap_and_plot(users_c, conv_c, users_v, conv_v, alpha_val=0.05):
    """
    Runs a parametric bootstrap by resampling from Binomial distributions.
    Returns (sim_c_pct, sim_v_pct, ci_lower_pct, ci_upper_pct).
    """
    rng = np.random.default_rng()
    p_c = safe_divide(conv_c, users_c)
    p_v = safe_divide(conv_v, users_v)
    sim_c = rng.binomial(users_c, p_c, BOOTSTRAP_SAMPLES) / users_c
    sim_v = rng.binomial(users_v, p_v, BOOTSTRAP_SAMPLES) / users_v
    diffs = sim_v - sim_c

    lower_pct = (alpha_val / 2) * 100
    upper_pct = (1 - alpha_val / 2) * 100
    ci_low = float(np.percentile(diffs, lower_pct))
    ci_high = float(np.percentile(diffs, upper_pct))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(diffs, bins=60, color="orange", edgecolor="black", alpha=0.7)
    ax.axvline(ci_low, color="red", linestyle="--", label=f"{(1-alpha_val)*100:.0f}% CI Lower")
    ax.axvline(ci_high, color="red", linestyle="--", label=f"{(1-alpha_val)*100:.0f}% CI Upper")
    ax.axvline(0, color="black", linestyle="-", linewidth=1.2, label="No effect")
    ax.set_title("Bootstrap Distribution of CR Difference (Variation − Control)")
    ax.set_xlabel("Difference in Conversion Rate")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)  # FIX: Prevent memory leak

    return sim_c * 100, sim_v * 100, ci_low * 100, ci_high * 100


def plot_box_plot_analysis(samples_c, samples_v):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        [samples_c, samples_v],
        patch_artist=True,
        widths=0.6,
        boxprops=dict(facecolor="skyblue"),
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Control Group", "Variation Group"])
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_title("Box Plot: Bootstrap Conversion Rate Distributions")
    st.pyplot(fig)
    plt.close(fig)  # FIX: Prevent memory leak


# -----------------------------------------------
# SIDEBAR — SETTINGS
# -----------------------------------------------
st.sidebar.markdown(
    f"""<div style="display:flex;align-items:center;">{ICON_SETTINGS}
    <h2 style="display:inline;font-size:1.5rem;margin-left:5px;">Settings</h2></div>""",
    unsafe_allow_html=True,
)

# --- Save & Load ---
st.sidebar.markdown(
    f"""<div style="display:flex;align-items:center;margin-top:10px;">{ICON_UPLOAD}
    <span style="font-size:1rem;font-weight:600;margin-left:5px;">Save & Load Analysis</span></div>""",
    unsafe_allow_html=True,
)

SAVE_KEYS = [
    "users_c", "conv_c", "rev_c", "prod_c",
    "users_v", "conv_v", "rev_v", "prod_v",
    "days", "conf_level",
    "s1_uc", "s1_cc", "s1_uv", "s1_cv",
    "s2_uc", "s2_cc", "s2_uv", "s2_cv",
]

with st.sidebar.expander("Manage Files", expanded=False):
    st.info(
        "Snapshot your experiment mid-run. Download on Day 7, re-upload on Day 14 to "
        "compare how Bayesian probability and risk evolve without re-entering data."
    )
    snapshot = {k: st.session_state.get(k) for k in SAVE_KEYS}
    st.download_button(
        label="Download Inputs (.json)",
        data=json.dumps(snapshot, indent=2),
        file_name="experiment_snapshot.json",
        mime="application/json",
    )
    uploaded = st.file_uploader("Load Snapshot", type=["json"])
    if uploaded is not None:
        try:
            loaded = json.load(uploaded)
            # Only restore known keys to prevent state pollution
            for k in SAVE_KEYS:
                if k in loaded:
                    st.session_state[k] = loaded[k]
            st.success("Snapshot loaded successfully!")
            st.rerun()
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            st.error(f"Could not load file: {e}")

# --- Confidence Level ---
confidence_level = st.sidebar.selectbox(
    "Confidence Level",
    ["95%", "90%", "99%"],
    index=0,
    key="conf_level",
    help=(
        "⚡ 90% — Fast decisions: colours, copy, low-risk UI tweaks.\n"
        "⚖️ 95% — Standard: features, flows, layouts.\n"
        "🐢 99% — High-stakes: pricing, checkout, backend algorithms."
    ),
)
alpha = {"90%": 0.10, "95%": 0.05, "99%": 0.01}[confidence_level]

st.sidebar.markdown("---")

# -----------------------------------------------
# SIDEBAR — EXPERIMENT PLANNING
# -----------------------------------------------
st.sidebar.markdown(
    f"""<div style="display:flex;align-items:center;">{ICON_CALENDAR}
    <h2 style="display:inline;font-size:1.5rem;margin-left:5px;">Experiment Planning</h2></div>""",
    unsafe_allow_html=True,
)

with st.sidebar.expander("Sample Size Calculator", expanded=False):
    st.caption(f"Plan required traffic at {confidence_level} confidence, 80% power.")

    plan_traffic = st.number_input("Traffic (Last 28 Days)", step=1000, key="p_traffic")
    plan_base_cr = st.number_input("Baseline Conversion Rate (%)", step=0.1, key="p_base_cr")
    plan_base_aov = st.number_input("Baseline AOV ($)", step=1.0, key="p_base_aov")
    plan_mde = st.number_input("Min Detectable Effect (%)", step=0.5, key="p_mde")
    volatility = st.selectbox(
        "Revenue Variance",
        ["Low (Subscription)", "Medium (Standard E-com)", "High (Whales/B2B)"],
        index=1,
        key="p_vol",
    )
    sd_multiplier = {"Low": 1.0, "Medium": 2.0, "High": 3.0}[volatility.split()[0]]

    if st.button("Calculate Duration"):
        if plan_base_cr <= 0 or plan_base_aov <= 0 or plan_traffic <= 0:
            st.error("Traffic, Baseline CR, and AOV must all be greater than 0.")
        else:
            daily_traffic = plan_traffic / 28
            p1 = plan_base_cr / 100
            p2 = p1 * (1 + plan_mde / 100)

            # Conversion rate sample size
            es_cr = proportion_effectsize(p1, p2)
            n_cr = NormalIndPower().solve_power(effect_size=es_cr, alpha=alpha, power=0.8, ratio=1)

            # Revenue (RPV) sample size
            est_sd = plan_base_aov * sd_multiplier
            es_rpv = safe_divide(plan_base_aov * plan_mde / 100, est_sd)
            if es_rpv > 0:
                n_rpv = TTestIndPower().solve_power(effect_size=es_rpv, alpha=alpha, power=0.8, ratio=1)
                n_rpv_visitors = safe_divide(n_rpv, p1)
                days_rpv = safe_divide(n_rpv_visitors * 2, daily_traffic)
            else:
                n_rpv_visitors, days_rpv = 0, 0

            days_cr = safe_divide(n_cr * 2, daily_traffic)

            st.markdown("---")
            st.write(f"**Daily Traffic:** {int(daily_traffic):,} users")
            st.info(f"**Conversion Rate:** {int(days_cr)} days needed ({int(n_cr):,} users/group)")
            fn = st.error if days_rpv > 60 else st.warning
            fn(f"**Revenue (RPV):** {int(days_rpv)} days needed ({int(n_rpv_visitors):,} users/group)")

st.sidebar.markdown("---")

# -----------------------------------------------
# SIDEBAR — DATA INPUTS
# -----------------------------------------------
st.sidebar.markdown(
    f"""<div style="display:flex;align-items:center;">{ICON_TROPHY}
    <h2 style="display:inline;font-size:1.5rem;margin-left:5px;">Enter Results</h2></div>""",
    unsafe_allow_html=True,
)
st.sidebar.caption("Input your experiment data below.")

st.sidebar.subheader("Control Group")
users_control = st.sidebar.number_input("Control Users", min_value=1, key="users_c")
conv_control = st.sidebar.number_input("Control Conversions", min_value=0, key="conv_c")
rev_control = st.sidebar.number_input("Control Revenue ($)", min_value=0.0, key="rev_c")
prod_control = st.sidebar.number_input("Control Products Sold", min_value=0, key="prod_c")

st.sidebar.markdown("---")

st.sidebar.subheader("Variation Group")
users_variation = st.sidebar.number_input("Variation Users", min_value=1, key="users_v")
conv_variation = st.sidebar.number_input("Variation Conversions", min_value=0, key="conv_v")
rev_variation = st.sidebar.number_input("Variation Revenue ($)", min_value=0.0, key="rev_v")
prod_variation = st.sidebar.number_input("Variation Products Sold", min_value=0, key="prod_v")

st.sidebar.markdown("---")
days_run = st.sidebar.number_input("Days Test Ran", min_value=1, key="days")


# -----------------------------------------------
# CORE METRIC CALCULATIONS
# -----------------------------------------------
rate_c = safe_divide(conv_control, users_control)
rate_v = safe_divide(conv_variation, users_variation)
uplift_cr = calculate_uplift(rate_c, rate_v)

aov_c = safe_divide(rev_control, conv_control)
aov_v = safe_divide(rev_variation, conv_variation)
uplift_aov = calculate_uplift(aov_c, aov_v)

rpv_c = safe_divide(rev_control, users_control)
rpv_v = safe_divide(rev_variation, users_variation)
uplift_rpv = calculate_uplift(rpv_c, rpv_v)

apo_c = safe_divide(prod_control, conv_control)
apo_v = safe_divide(prod_variation, conv_variation)
uplift_apo = calculate_uplift(apo_c, apo_v)

apu_c = safe_divide(prod_control, users_control)
apu_v = safe_divide(prod_variation, users_variation)
uplift_apu = calculate_uplift(apu_c, apu_v)

# Statistical tests
z_stat, p_value_z = proportions_ztest(
    [conv_control, conv_variation],
    [users_control, users_variation],
)
srm_stat, p_value_srm = perform_srm_test([users_control, users_variation])


# -----------------------------------------------
# MAIN DASHBOARD
# -----------------------------------------------
st.title("A/B Test Analyzer v1.8.0c")

render_header(ICON_BAR_CHART, "Results Summary")

st.subheader("1. Primary KPIs")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Conversion Rate", f"{rate_v*100:.2f}%", f"{uplift_cr:+.2f}%")
col2.metric("Revenue Per Visitor", f"${rpv_v:.2f}", f"{uplift_rpv:+.2f}%")
col3.metric("Avg Order Value", f"${aov_v:.2f}", f"{uplift_aov:+.2f}%")
if p_value_z <= alpha:
    col4.success(f"CR Sig: YES (p={p_value_z:.4f})")
else:
    col4.info(f"CR Sig: NO (p={p_value_z:.4f})")

st.subheader("2. Product Velocity")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Products / Order", f"{apo_v:.2f}", f"{uplift_apo:+.2f}%")
col2.metric("Avg Products / User", f"{apu_v:.2f}", f"{uplift_apu:+.2f}%")
col4.caption("Higher 'Products / Order' means users are building bigger baskets.")

st.markdown("---")

st.subheader("3. Executive Interpretation")
if uplift_cr > 0 and uplift_aov < 0:
    st.warning("Trade-off Detected: Variation drives MORE orders but SMALLER baskets.")
elif uplift_cr < 0 and uplift_aov > 0:
    st.warning("Trade-off Detected: Variation drives FEWER orders but BIGGER baskets.")
else:
    st.success("Clean Result: Conversion and order value trends are consistent.")

if uplift_rpv >= 0:
    st.write(f"**Financial Impact:** Variation generates **${rpv_v - rpv_c:.2f} more** per visitor.")
else:
    st.write(f"**Financial Impact:** Variation generates **${rpv_c - rpv_v:.2f} less** per visitor.")

st.subheader("4. Health Checks")
col1, col2 = st.columns(2)
with col1:
    if p_value_srm < 0.01:
        st.error(f"⚠️ SRM DETECTED (p={p_value_srm:.4f}) — traffic split is uneven. Results may be invalid.")
    else:
        st.success(f"✅ SRM PASSED (p={p_value_srm:.4f})")
with col2:
    if days_run < 7:
        st.error(f"⚠️ Too Short ({days_run} days) — results are unreliable.")
    elif days_run < 14:
        st.warning(f"⚠️ Short Duration ({days_run} days) — watch for novelty effects.")
    else:
        st.success(f"✅ Duration OK ({days_run} days)")

st.markdown("---")

# -----------------------------------------------
# SIMPSON'S PARADOX DETECTOR
# -----------------------------------------------
render_header(ICON_PIE, "Advanced: Simpson's Paradox Detector", level=3)

with st.expander("Expand to check segments (e.g. Mobile vs Desktop)", expanded=False):
    st.caption("Detects when segment-level trends contradict the aggregate result.")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("Segment 1 (e.g. Mobile)")
        s1_uc = st.number_input("Control Users", min_value=0, key="s1_uc")
        s1_cc = st.number_input("Control Conv.", min_value=0, key="s1_cc")
        s1_uv = st.number_input("Var Users", min_value=0, key="s1_uv")
        s1_cv = st.number_input("Var Conv.", min_value=0, key="s1_cv")
    with col_s2:
        st.subheader("Segment 2 (e.g. Desktop)")
        s2_uc = st.number_input("Control Users", min_value=0, key="s2_uc")
        s2_cc = st.number_input("Control Conv.", min_value=0, key="s2_cc")
        s2_uv = st.number_input("Var Users", min_value=0, key="s2_uv")
        s2_cv = st.number_input("Var Conv.", min_value=0, key="s2_cv")

    if st.button("Check for Paradox"):
        seg1 = {"users_c": s1_uc, "conv_c": s1_cc, "users_v": s1_uv, "conv_v": s1_cv}
        seg2 = {"users_c": s2_uc, "conv_c": s2_cc, "users_v": s2_uv, "conv_v": s2_cv}
        is_paradox, up1, up2, up_agg = check_simpsons_paradox(seg1, seg2)

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Segment 1 Uplift", f"{up1:.2f}%")
        m2.metric("Segment 2 Uplift", f"{up2:.2f}%")
        m3.metric("Combined Uplift", f"{up_agg:.2f}%")

        if is_paradox:
            st.error("🚨 SIMPSON'S PARADOX DETECTED! Segment trends contradict the aggregate result.")
            st.warning("Trust the segment data. The aggregate result is misleading due to traffic imbalances.")
        else:
            st.success("✅ No Paradox Detected. Trends are consistent across segments.")

st.markdown("---")

# -----------------------------------------------
# DEEP DIVE TABS
# -----------------------------------------------
render_header(ICON_BRAIN, "Deep Dive Analysis")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "Smart Analysis", "AI Analysis", "Stopping & Sequential",
    "Strategic Matrix", "Product Metrics", "Revenue Charts",
    "CR Comparison", "Bayesian", "Bootstrap", "Box Plot",
])

# --- TAB 1: SMART ANALYSIS ---
with tab1:
    st.markdown("### Auto-Analyst Report")
    st.info("Instant rule-based analysis — no API key required.")
    user_hypothesis = st.text_area(
        "Hypothesis:", placeholder="We believed that...", height=70, key="hyp_smart"
    )
    if st.button("Generate Smart Report"):
        ci_low_abs, ci_high_abs = proportion_confint(
            conv_variation, users_variation, alpha=alpha, method="normal"
        )
        diff_ci_low = (ci_low_abs - rate_c) * 100
        diff_ci_high = (ci_high_abs - rate_c) * 100
        prob_v_wins, loss_v, loss_c = calculate_bayesian_risk(
            conv_control + 1, users_control - conv_control + 1,
            conv_variation + 1, users_variation - conv_variation + 1,
        )
        metrics_payload = {
            "days": days_run, "users_c": users_control, "users_v": users_variation,
            "p_srm": p_value_srm, "cr_c": rate_c * 100, "cr_v": rate_v * 100,
            "uplift_cr": uplift_cr, "p_cr": p_value_z,
            "aov_c": aov_c, "aov_v": aov_v, "uplift_aov": uplift_aov,
            "rpv_c": rpv_c, "rpv_v": rpv_v, "uplift_rpv": uplift_rpv,
            "ci_low": diff_ci_low, "ci_high": diff_ci_high,
            "uplift_apo": uplift_apo,
            "prob_v_wins": prob_v_wins * 100,
            "loss_v": loss_v * 100, "loss_c": loss_c * 100,
            "alpha": alpha,
        }
        st.markdown("---")
        st.markdown(generate_smart_analysis(user_hypothesis, metrics_payload, alpha))

# --- TAB 2: AI ANALYSIS ---
with tab2:
    st.markdown("### AI-Powered Analysis")
    col_prov, col_key = st.columns(2)
    with col_prov:
        ai_provider = st.selectbox("Provider", ["OpenAI (GPT-4o)", "DeepSeek (R1)"])
    with col_key:
        api_key_input = st.text_input("API Key", type="password")
    user_hypothesis_ai = st.text_area(
        "Hypothesis:", placeholder="We believed that...", height=100, key="hyp_ai"
    )
    if st.button("Generate AI Analysis"):
        ci_low_abs, ci_high_abs = proportion_confint(
            conv_variation, users_variation, alpha=alpha, method="normal"
        )
        diff_ci_low = (ci_low_abs - rate_c) * 100
        diff_ci_high = (ci_high_abs - rate_c) * 100
        prob_v_wins, loss_v, loss_c = calculate_bayesian_risk(
            conv_control + 1, users_control - conv_control + 1,
            conv_variation + 1, users_variation - conv_variation + 1,
        )
        metrics_payload = {
            "days": days_run, "users_c": users_control, "users_v": users_variation,
            "p_srm": p_value_srm, "cr_c": rate_c * 100, "cr_v": rate_v * 100,
            "uplift_cr": uplift_cr, "p_cr": p_value_z,
            "aov_c": aov_c, "aov_v": aov_v, "uplift_aov": uplift_aov,
            "rpv_c": rpv_c, "rpv_v": rpv_v, "uplift_rpv": uplift_rpv,
            "ci_low": diff_ci_low, "ci_high": diff_ci_high,
            "prob_v_wins": prob_v_wins * 100,
            "loss_v": loss_v * 100, "loss_c": loss_c * 100,
        }
        provider_name = "DeepSeek" if "DeepSeek" in ai_provider else "OpenAI"
        with st.spinner(f"Connecting to {provider_name}..."):
            ai_result = get_ai_analysis(
                api_key_input, user_hypothesis_ai, metrics_payload,
                provider=provider_name, conf_level=confidence_level,
            )
        st.markdown(f"### Analysis: {user_hypothesis_ai or 'A/B Test'}")
        st.markdown("---")
        st.markdown(ai_result)  # FIX: Result is now displayed

# --- TAB 3: STOPPING & SEQUENTIAL ---
with tab3:
    st.markdown("### Stopping Rules & Sequential Testing")
    prob_v_wins, loss_v, loss_c = calculate_bayesian_risk(
        conv_control + 1, users_control - conv_control + 1,
        conv_variation + 1, users_variation - conv_variation + 1,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Probability Variation is Best", f"{prob_v_wins*100:.1f}%")
    c2.metric("Risk if You Switch", f"{loss_v*100:.5f}%")
    c3.metric("Risk if You Stay", f"{loss_c*100:.5f}%")

    st.markdown("---")
    st.subheader("Sequential Testing / Peeking Penalty")
    st.caption(
        "Checking results repeatedly inflates your false positive rate. "
        "This calculator penalises your alpha accordingly. "
        "Note: this uses a custom log-based approximation — for production use, "
        "consider O'Brien-Fleming or Pocock alpha-spending functions."
    )
    peeks = st.number_input("How many times have you checked results?", min_value=1, value=1)
    # Custom approximation: alpha is penalised log-linearly as peeks increase
    adjusted_alpha = alpha * np.log(1 + (np.e - 1) / peeks)
    st.write(f"Standard threshold: **{alpha}** → Adjusted threshold: **{adjusted_alpha:.5f}**")
    if p_value_z < adjusted_alpha:
        st.success(f"✅ SIGNIFICANT after peeking penalty (p={p_value_z:.4f})")
    else:
        st.error(f"❌ NOT SIGNIFICANT after peeking penalty (p={p_value_z:.4f})")

# --- TAB 4: STRATEGIC MATRIX ---
with tab4:
    plot_strategic_matrix(rate_c * 100, aov_c, rate_v * 100, aov_v)

# --- TAB 5: PRODUCT METRICS ---
with tab5:
    c1, c2 = st.columns(2)
    with c1:
        plot_metric_comparison("Avg Products / Order", apo_c, apo_v)
    with c2:
        plot_metric_comparison("Avg Products / User", apu_c, apu_v)

# --- TAB 6: REVENUE CHARTS ---
with tab6:
    c1, c2 = st.columns(2)
    with c1:
        plot_metric_comparison("Revenue Per Visitor", rpv_c, rpv_v, "$")
    with c2:
        plot_metric_comparison("Avg Order Value", aov_c, aov_v, "$")

# --- TAB 7: CR COMPARISON ---
with tab7:
    plot_metric_comparison("Conversion Rate", rate_c * 100, rate_v * 100)

# --- TAB 8: BAYESIAN PDFs ---
with tab8:
    plot_bayesian_pdfs(
        conv_control + 1, users_control - conv_control + 1,
        conv_variation + 1, users_variation - conv_variation + 1,
    )

# --- TAB 9 & 10: BOOTSTRAP + BOX PLOT ---
# FIX: Both tabs share bootstrap data. We compute it once in session_state
# so tab10 is never dependent on tab9 having rendered first.
with tab9:
    st.markdown("### Bootstrap Confidence Interval")
    samples_c, samples_v, boot_ci_low, boot_ci_high = run_bootstrap_and_plot(
        users_control, conv_control, users_variation, conv_variation, alpha_val=alpha
    )
    # Store results so tab10 can access them safely
    st.session_state["_boot_samples_c"] = samples_c
    st.session_state["_boot_samples_v"] = samples_v
    st.write(
        f"**{confidence_level} CI on CR Difference (Variation − Control):** "
        f"{boot_ci_low:.2f}% to {boot_ci_high:.2f}%"
    )

with tab10:
    st.markdown("### Box Plot: Bootstrap Distributions")
    # FIX: Retrieve from session_state — safe even if tab9 hasn't rendered this session
    if "_boot_samples_c" in st.session_state and "_boot_samples_v" in st.session_state:
        plot_box_plot_analysis(
            st.session_state["_boot_samples_c"],
            st.session_state["_boot_samples_v"],
        )
    else:
        st.info("Open the **Bootstrap** tab first to generate the data, then return here.")
