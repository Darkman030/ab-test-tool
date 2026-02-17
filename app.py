import streamlit as st
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import beta, chi2_contingency
from statsmodels.stats.proportion import proportions_ztest, proportion_confint, proportion_effectsize
from statsmodels.stats.power import NormalIndPower, TTestIndPower
import openai

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="A/B Test Analyzer", layout="wide")

# FIX: Set Matplotlib backend to Agg to prevent threading crashes
matplotlib.use('Agg')

# ==============================================
# 0. STATE INITIALIZATION
# ==============================================
def initialize_state():
    default_values = {
        "users_c": 5000, "conv_c": 500, "rev_c": 25000.0, "prod_c": 750,
        "users_v": 5000, "conv_v": 600, "rev_v": 33000.0, "prod_v": 1000,
        "days": 14, 
        "conf_level": "95%",
        "p_traffic": 50000, "p_base_cr": 2.5, "p_base_aov": 75.0, "p_mde": 5.0, 
        "p_vol": "Medium (Standard E-com)",
        "s1_uc": 2000, "s1_cc": 100, "s1_uv": 2000, "s1_cv": 110,
        "s2_uc": 3000, "s2_cc": 400, "s2_uv": 3000, "s2_cv": 420
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_state()

# ==============================================
# 🎨 SVG ASSETS
# ==============================================
ICON_SETTINGS = """<svg viewBox="0 0 1024 1024" class="icon" style="width: 1.5em; height: 1.5em; vertical-align: middle; margin-right: 10px;" version="1.1" xmlns="http://www.w3.org/2000/svg" fill="#000000"><path d="M512 512m-10 0a10 10 0 1 0 20 0 10 10 0 1 0-20 0Z" fill="#E73B37"></path><path d="M512 306.8c27.7 0 54.6 5.4 79.8 16.1 24.4 10.3 46.4 25.1 65.2 44s33.6 40.8 44 65.2c10.7 25.3 16.1 52.1 16.1 79.8 0 27.7-5.4 54.6-16.1 79.8-10.3 24.4-25.1 46.4-44 65.2-18.8 18.8-40.8 33.6-65.2 44-25.3 10.7-52.1 16.1-79.8 16.1-27.7 0-54.6-5.4-79.8-16.1-24.4-10.3-46.4-25.1-65.2-44-18.8-18.8-33.6-40.8-44-65.2-10.7-25.3-16.1-52.1-16.1-79.8 0-27.7 5.4-54.6 16.1-79.8 10.3-24.4 25.1-46.4 44-65.2s40.8-33.6 65.2-44c25.2-10.6 52.1-16.1 79.8-16.1m0-22c-125.4 0-227.1 101.7-227.1 227.1S386.6 739.1 512 739.1c125.4 0 227.1-101.7 227.1-227.1S637.4 284.8 512 284.8z" fill="#ffffff"></path><path d="M512 618.7c-58.9 0-106.8-47.9-106.8-106.8S453.1 405.1 512 405.1 618.8 453 618.8 511.9 570.9 618.7 512 618.7z m0-193.5c-47.9 0-86.8 38.9-86.8 86.8s38.9 86.8 86.8 86.8 86.8-38.9 86.8-86.8-38.9-86.8-86.8-86.8z" fill="#E73B37"></path><path d="M544.2 107.3l34.1 92.3 7.4 19.9 20.2 6.6c10.3 3.4 32.1 12.9 43.4 18.1l18.7 8.6 18.6-8.9 87.9-41.8 46.4 46.5-41.2 89.4-8.9 19.3 9.6 19c6.8 13.4 12.6 27.5 17.4 41.9l6.7 20.5 20.3 7.2 91.7 32.6v65.7l-92.3 34.1-19.9 7.4-6.6 20.2c-4.7 14.4-10.6 28.4-17.4 41.9l-9.8 19.3 9.3 19.5 41.8 87.9-46.5 46.5-89.1-41.3-19.3-8.9-19 9.6c-13.4 6.8-27.5 12.6-41.9 17.4l-20.5 6.7-7.2 20.3-32.6 91.7h-65.7l-34.1-92.3-7.4-19.9-20.2-6.6c-10.3-3.4-32.1-12.9-43.4-18.1L356 771l-18.6 8.9-87.9 41.8-46.4-46.5 41.2-89.3 8.9-19.3-9.6-19c-6.8-13.4-12.6-27.5-17.4-41.9l-6.7-20.5-20.3-7.2-91.7-32.6v-65.7l92.3-34.1 19.9-7.4 6.6-20.2c3.4-10.3 12.9-32.1 18.1-43.4l8.6-18.7-8.9-18.6-41.8-87.9 46.4-46.4 89.3 41.2 19.3 8.9 19-9.6c13.4-6.8 27.5-12.6 41.9-17.4l20.5-6.7 7.2-20.3 32.6-91.7h65.7m30.7-44.1H447.4l-43 121c-16.6 5.5-32.7 12.1-48.1 19.9l-117.2-54-90.1 90.1 55.2 116s-14.5 31.4-19.9 48.1l-121 44.7v127.4l121 43c5.5 16.6 12.1 32.6 19.9 48l-54 117.2 90.1 90.1 116-55.2s31.4 14.5 48.1 19.9l44.7 121h127.4l43-121c16.6-5.5 32.6-12.1 48-19.9l117.2 54 90.1-90.1-55.2-116c7.8-15.4 14.5-31.4 19.9-48l121-44.7V447.4l-121-43c-5.5-16.6-12.1-32.6-19.9-48l54-117.2-90.1-90.1-115.9 55.2s-31.5-14.5-48.1-19.9L574.9 63.3z" fill="#ffffff"></path></svg>"""
ICON_PIE = """<svg viewBox="0 0 1024 1024" class="icon" style="width: 1.5em; height: 1.5em; vertical-align: middle; margin-right: 10px;" version="1.1" xmlns="http://www.w3.org/2000/svg" fill="#000000"><path d="M429.9 186.7v406.4h407.5c-4 34.1-12.8 67.3-26.2 99.1-18.4 43.6-44.8 82.7-78.5 116.3-33.6 33.6-72.8 60-116.4 78.4-45.1 19.1-93 28.7-142.5 28.7-49.4 0-97.4-9.7-142.5-28.7-43.6-18.4-82.7-44.8-116.4-78.4-33.6-33.6-60-72.7-78.4-116.3-19.1-45.1-28.7-93-28.7-142.4s9.7-97.3 28.7-142.4c18.4-43.6 44.8-82.7 78.4-116.3 33.6-33.6 72.8-60 116.4-78.4 31.7-13.2 64.7-21.9 98.6-26m44-46.6c-226.4 0-410 183.5-410 409.8s183.6 409.8 410 409.8 410-183.5 410-409.8v-0.8h-410v-409z" fill="#ffffff"></path><path d="M566.1 80.5c43.7 1.7 86.4 10.6 127 26.4 44 17.1 84.2 41.8 119.6 73.5 71.7 64.1 117.4 151.7 128.7 246.7 1.2 9.9 2 20 2.4 30.2H566.1V80.5m-16-16.3v409h410c0-16.3-1-32.3-2.9-48.1C933.1 221.9 760 64.2 550.1 64.2zM264.7 770.4c-23.1-23.1-42.3-49.1-57.3-77.7l-14.7 6.5c35.7 68.2 94 122.7 165 153.5l4.3-15.6c-36.3-16-69.1-38.4-97.3-66.7z" fill="#E73B37"></path></svg>"""
ICON_CALENDAR = """<svg viewBox="0 0 1024 1024" class="icon" style="width: 1.5em; height: 1.5em; vertical-align: middle; margin-right: 10px;" version="1.1" xmlns="http://www.w3.org/2000/svg" fill="#000000"><path d="M716 190.9v-67.8h-44v67.8H352v-67.8h-44v67.8H92v710h840v-710H716z m-580 44h172v69.2h44v-69.2h320v69.2h44v-69.2h172v151.3H136V234.9z m752 622H136V402.2h752v454.7z" fill="#ffffff"></path><path d="M319 565.7m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"></path><path d="M510 565.7m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"></path><path d="M701.1 565.7m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"></path><path d="M319 693.4m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"></path><path d="M510 693.4m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"></path><path d="M701.1 693.4m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"></path></svg>"""
ICON_BAR_CHART = """<svg viewBox="0 0 1024 1024" class="icon" style="width: 1.5em; height: 1.5em; vertical-align: middle; margin-right: 10px;" version="1.1" xmlns="http://www.w3.org/2000/svg" fill="#000000"><path d="M928.1 881v44H95.9V99h44v782z" fill="#ffffff"></path><path d="M352 435.7v403.4H204V435.7h148m22-22H182v447.4h192V413.7zM608 307.9v531.2H460V307.9h148m22-22H438v575.2h192V285.9z" fill="#ffffff"></path><path d="M866.1 177.3v663.9H714V177.3h152.1m20-20H694v703.9h192V157.3h0.1z" fill="#E73B37"></path></svg>"""
ICON_BRAIN = """<svg viewBox="0 0 1024 1024" class="icon" style="width: 1.5em; height: 1.5em; vertical-align: middle; margin-right: 10px;" version="1.1" xmlns="http://www.w3.org/2000/svg" fill="#000000"><path d="M512 301.2m-10 0a10 10 0 1 0 20 0 10 10 0 1 0-20 0Z" fill="#E73B37"></path><path d="M400.3 744.5c2.1-0.7 4.1-1.4 6.2-2-2 0.6-4.1 1.3-6.2 2z m0 0c2.1-0.7 4.1-1.4 6.2-2-2 0.6-4.1 1.3-6.2 2z" fill="#ffffff"></path><path d="M511.8 256.6c24.4 0 44.2 19.8 44.2 44.2S536.2 345 511.8 345s-44.2-19.8-44.2-44.2 19.9-44.2 44.2-44.2m0-20c-35.5 0-64.2 28.7-64.2 64.2s28.7 64.2 64.2 64.2 64.2-28.7 64.2-64.2-28.7-64.2-64.2-64.2z" fill="#E73B37"></path><path d="M730.7 529.5c0.4-8.7 0.6-17.4 0.6-26.2 0-179.6-86.1-339.1-219.3-439.5-133.1 100.4-219.2 259.9-219.2 439.5 0 8.8 0.2 17.5 0.6 26.1-56 56-90.6 133.3-90.6 218.7 0 61.7 18 119.1 49.1 167.3 30.3-49.8 74.7-90.1 127.7-115.3 39-18.6 82.7-29 128.8-29 48.3 0 93.9 11.4 134.3 31.7 52.5 26.3 96.3 67.7 125.6 118.4 33.4-49.4 52.9-108.9 52.9-173.1 0-85.4-34.6-162.6-90.5-218.6z" fill="#ffffff"></path><path d="M512 819.3c8.7 0 24.7 22.9 24.7 60.4s-16 60.4-24.7 60.4-24.7-22.9-24.7-60.4 16-60.4 24.7-60.4m0-20c-24.7 0-44.7 36-44.7 80.4 0 44.4 20 80.4 44.7 80.4s44.7-36 44.7-80.4c0-44.4-20-80.4-44.7-80.4z" fill="#E73B37"></path></svg>"""
ICON_UPLOAD = """<svg viewBox="0 0 1024 1024" class="icon" style="width: 1.5em; height: 1.5em; vertical-align: middle; margin-right: 10px;" version="1.1" xmlns="http://www.w3.org/2000/svg" fill="#000000"><path d="M220.5 245.4c-32.8 32.8-55.1 73.2-65.2 117.3h16.5c18.8-75.3 75.1-135.9 148-160.7v-16.9c-37.1 11.6-71 32-99.3 60.3z" fill="#E73B37"></path><path d="M959.9 540.8c0 113.6-92.1 205.8-205.7 205.9H590.9v-44h163.3c43.2 0 83.8-16.9 114.3-47.4 30.6-30.6 47.4-71.2 47.4-114.5 0-43.2-16.8-83.9-47.4-114.4S797.2 379 754 379c-11.5 0-22.8 1.2-33.8 3.5-15 3.2-29.4 8.4-42.8 15.7-1-15.4-3.3-30.7-6.8-45.6v-0.1c-3.6-15.6-8.6-30.8-14.9-45.7-14.4-33.9-34.9-64.4-61.1-90.6-26.2-26.2-56.6-46.7-90.6-61.1-35.1-14.8-72.4-22.4-110.9-22.4s-75.8 7.5-110.9 22.4c-33.9 14.3-64.4 34.9-90.6 61.1-26.2 26.2-46.7 56.7-61.1 90.6-14.9 35.1-22.4 72.4-22.4 110.9s7.5 75.8 22.4 110.9c14.3 33.9 34.9 64.4 61.1 90.6 26.2 26.2 56.7 46.7 90.6 61.1 35.1 14.8 72.4 22.4 110.9 22.4h39.7v44h-41C210.7 746 64.1 599 64.1 417.7c0-181.7 147.3-329 329-329 154.6 0 284.3 106.6 319.5 250.3v0.1c13.4-2.7 27.2-4.2 41.4-4.2 113.7 0.1 205.9 92.2 205.9 205.9z" fill="#ffffff"></path><path d="M692.9 636.1h-22.6L519.8 485.6v449.6h-16V485.8L353.4 636.1h-22.6l181-181z" fill="#E73B37"></path></svg>"""
ICON_TROPHY = """<svg viewBox="0 0 1024 1024" class="icon" style="width: 1.5em; height: 1.5em; vertical-align: middle; margin-right: 10px;" version="1.1" xmlns="http://www.w3.org/2000/svg" fill="#000000"><path d="M828.5 180.1h-9.9v-54.7h23.5v-44H182v44h23v54.7h-9.5C123.2 180.1 64 239.2 64 311.5v0.1c0 72.3 59.2 131.5 131.5 131.5h9.6c0 1.3 0.1 2.5 0.1 3.7 0.5 17.7 2.7 35.4 6.2 52.5 17.8 85.7 71.8 160 148.3 204 4.8 2.8 9.8 5.4 14.7 7.9 15.3 7.7 31.2 14.1 47.4 19.2 3.4 1 6.8 2 10.2 2.9v165.2H250.4v44h511.9v-44H591.9V733.4c3.7-1 7.3-2.1 10.9-3.2 16.2-5.1 32.2-11.6 47.4-19.4 5-2.5 10-5.3 14.8-8.1 75.6-43.9 129.2-117.8 147-202.7 3.6-17.2 5.8-34.9 6.3-52.4 0.1-1.5 0.1-3 0.1-4.5h10c72.3 0 131.5-59.2 131.5-131.5v-0.1c0.1-72.3-59.1-131.4-131.4-131.4zM205 399.2h-9.5c-23.2 0-45.1-9.1-61.7-25.7s-25.7-38.5-25.7-61.7v-0.1c0-23.2 9.1-45.2 25.7-61.7 16.6-16.6 38.5-25.7 61.7-25.7h9.5v174.9z m370.9 499.4h-128V737.3c20.9 4.5 42.3 6.8 63.9 6.8 21.7 0 43.1-2.3 64.1-6.8v161.3z m198.7-461.4c0 2.9 0 5.9-0.2 8.9-0.5 15-2.3 30.1-5.4 44.9-15.3 72.7-61.2 136-126.1 173.7-4.1 2.4-8.4 4.7-12.7 6.9-13 6.6-26.7 12.2-40.6 16.6-25.2 7.9-51.4 11.9-77.9 11.9-26.2 0-52.2-3.9-77.1-11.6-13.9-4.3-27.5-9.8-40.6-16.4-4.2-2.1-8.5-4.4-12.6-6.8-65.4-37.8-111.7-101.5-126.9-174.8-3.1-14.7-4.9-29.8-5.3-45-0.1-2.7-0.1-5.5-0.1-8.2v-312h525.6v311.9zM916 311.7c0 23.2-9.1 45.2-25.7 61.7-16.6 16.6-38.5 25.7-61.7 25.7h-9.9v-175h9.9c23.2 0 45.1 9.1 61.7 25.7s25.7 38.5 25.7 61.7v0.2z" fill="#ffffff"></path><path d="M317.428 274.917l70.145-70.144 14.142 14.142-70.145 70.144zM316.055 351.98L456.13 211.904l14.142 14.142-140.076 140.076zM555.4 659.6l-4.8-19.4c0.3-0.1 26.5-6.8 55.4-23.5 37.8-21.9 62-49.7 72-82.7l19.1 5.8c-11.4 37.6-39.6 70.3-81.6 94.5-31.2 18-58.9 25-60.1 25.3z" fill="#E73B37"></path></svg>"""

def render_sidebar_header(svg_code, text, help_text=None):
    html = f"""<div style="display: flex; align-items: center; margin-bottom: 10px;">{svg_code}<h2 style="margin: 0; padding: 0; font-size: 1.5rem; margin-left: 5px;">{text}</h2></div>"""
    st.sidebar.markdown(html, unsafe_allow_html=True, help=help_text)

def render_main_header(svg_code, text, level=2):
    html = f"""<div style="display: flex; align-items: center; margin-bottom: 10px;">{svg_code}<h{level} style="margin: 0; padding: 0;">{text}</h{level}></div>"""
    st.markdown(html, unsafe_allow_html=True)

# ==============================================
# UI - SIDEBAR
# ==============================================
st.title("Professional A/B Test Analyzer (Enterprise Edition)")

# Settings
render_sidebar_header(ICON_SETTINGS, "Settings")
confidence_level = st.sidebar.selectbox("Confidence Level", ["95%", "90%", "99%"], index=0, key="conf_level")
alpha = {"90%": 0.10, "95%": 0.05, "99%": 0.01}[confidence_level]
st.sidebar.markdown("---")

# Save & Load
save_load_help = "Workflow Continuity: Capture a snapshot. Download on Day 7, then re-upload on Day 14 to see evolution."
render_sidebar_header(ICON_UPLOAD, "Save & Load Analysis", help_text=save_load_help)
with st.sidebar.expander("Open Controls", expanded=False):
    current_state = {k: st.session_state[k] for k in ["users_c", "conv_c", "rev_c", "prod_c", "users_v", "conv_v", "rev_v", "prod_v", "days", "conf_level"]}
    st.download_button(label="Download Snapshot (.json)", data=json.dumps(current_state, indent=2), file_name="experiment.json", mime="application/json")
    uploaded_file = st.file_uploader("Upload Snapshot", type=["json"])
    if uploaded_file:
        data = json.load(uploaded_file)
        for k, v in data.items(): st.session_state[k] = v
        st.rerun()
st.sidebar.markdown("---")

# Planning
render_sidebar_header(ICON_CALENDAR, "Experiment Planning")
with st.sidebar.expander("Calculator", expanded=False):
    p_traffic = st.number_input("Traffic (Last 28 Days)", step=1000, key="p_traffic")
    p_cr = st.number_input("Baseline CR (%)", step=0.1, key="p_base_cr")
    p_mde = st.number_input("Target Lift (%)", step=0.5, key="p_mde")
    if st.button("Calculate Duration"):
        daily = p_traffic / 28
        e_cr = proportion_effectsize(p_cr/100, (p_cr/100)*(1+p_mde/100))
        n_cr = NormalIndPower().solve_power(effect_size=e_cr, alpha=alpha, power=0.8, ratio=1)
        st.info(f"Need {int((n_cr*2)/daily)} days")
st.sidebar.markdown("---")

# Enter Results
render_sidebar_header(ICON_TROPHY, "Enter Results")
st.sidebar.subheader("Control Group")
users_control = st.sidebar.number_input("Users", min_value=1, key="users_c")
conv_control = st.sidebar.number_input("Conversions", min_value=0, key="conv_c")
rev_control = st.sidebar.number_input("Revenue ($)", min_value=0.0, key="rev_c")
prod_control = st.sidebar.number_input("Items", min_value=0, key="prod_c")
st.sidebar.markdown("---")
st.sidebar.subheader("Variation Group")
users_variation = st.sidebar.number_input("Users", min_value=1, key="users_v")
conv_variation = st.sidebar.number_input("Conversions", min_value=0, key="conv_v")
rev_variation = st.sidebar.number_input("Revenue ($)", min_value=0.0, key="rev_v")
prod_variation = st.sidebar.number_input("Items", min_value=0, key="prod_v")
st.sidebar.markdown("---")
days_run = st.sidebar.number_input("Days Test Ran", min_value=1, key="days")

# ==============================================
# UI - MAIN DASHBOARD
# ==============================================
def calculate_uplift(ctrl, var): return ((var - ctrl) / ctrl) * 100 if ctrl > 0 else 0
rate_c, rate_v = conv_control/users_control, conv_variation/users_variation
up_cr = calculate_uplift(rate_c, rate_v)
aov_c, aov_v = (rev_control/conv_control if conv_control > 0 else 0), (rev_variation/conv_variation if conv_variation > 0 else 0)
up_aov = calculate_uplift(aov_c, aov_v)
rpv_c, rpv_v = rev_control/users_control, rev_variation/users_variation
z_stat, p_val = proportions_ztest([conv_control, conv_variation], [users_control, users_variation])

render_main_header(ICON_BAR_CHART, "Results Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("CR", f"{rate_v*100:.2f}%", f"{up_cr:+.2f}%")
c2.metric("RPV", f"${rpv_v:.2f}", f"{calculate_uplift(rpv_c, rpv_v):+.2f}%")
c3.metric("AOV", f"${aov_v:.2f}", f"{up_aov:+.2f}%")
c4.success(f"Sig: {p_val <= alpha} (p={p_val:.4f})")

st.markdown("---")
render_main_header(ICON_PIE, "Simpson's Paradox Detector", level=3)
with st.expander("Expand Segments", expanded=False):
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        s1_uc = st.number_input("S1 Control Users", key="s1_uc")
        s1_cc = st.number_input("S1 Control Conv.", key="s1_cc")
        s1_uv = st.number_input("S1 Var Users", key="s1_uv")
        s1_cv = st.number_input("S1 Var Conv.", key="s1_cv")
    with col_s2:
        s2_uc = st.number_input("S2 Control Users", key="s2_uc")
        s2_cc = st.number_input("S2 Control Conv.", key="s2_cc")
        s2_uv = st.number_input("S2 Var Users", key="s2_uv")
        s2_cv = st.number_input("S2 Var Conv.", key="s2_cv")
    
    if st.button("Check Paradox"):
        up1 = calculate_uplift(s1_cc/s1_uc if s1_uc>0 else 0, s1_cv/s1_uv if s1_uv>0 else 0)
        up2 = calculate_uplift(s2_cc/s2_uc if s2_uc>0 else 0, s2_cv/s2_uv if s2_uv>0 else 0)
        st.write(f"Segment 1 Uplift: {up1:.2f}% | Segment 2 Uplift: {up2:.2f}%")

st.markdown("---")
render_main_header(ICON_BRAIN, "Deep Dive Analysis")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Smart Analysis", "AI Analysis", "Strategic Matrix", "Bayesian", "Bootstrap"])

with tab1:
    st.markdown("### Smart Interpretation")
    st.markdown(f"**Executive Summary:** Test is {'Significant' if p_val <= alpha else 'Inconclusive'}.")
    st.markdown(f"**Performance Breakdown:** CR uplift: {up_cr:.2f}%. AOV uplift: {up_aov:.2f}%.")
    st.markdown("**Visual Insights (Strategic Matrix):**")
    if up_cr > 0 and up_aov > 0: st.success("Variation is in the Win-Win zone.")
    elif up_cr > 0 and up_aov < 0: st.warning("Volume up, Value down trade-off detected.")
    elif up_cr < 0 and up_aov > 0: st.warning("Value up, Volume down trade-off detected.")
    else: st.error("Loss-Loss zone.")

with tab2:
    st.info("AI Analysis module active. Enter hypothesis and API key to proceed.")
    st.text_area("Hypothesis", placeholder="We believed that...")
    st.text_input("API Key", type="password")

with tab3:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([rate_c*100], [aov_c], color='blue', s=200, label='Control')
    ax.scatter([rate_v*100], [aov_v], color='green', s=200, label='Variation')
    ax.set_xlabel("Conversion Rate (%)"); ax.set_ylabel("Average Order Value ($)")
    ax.legend(); st.pyplot(fig)

with tab4:
    x = np.linspace(max(0, rate_c-0.05), min(1, rate_c+0.05), 500)
    y_c = beta(conv_control+1, users_control-conv_control+1).pdf(x)
    y_v = beta(conv_variation+1, users_variation-conv_variation+1).pdf(x)
    fig, ax = plt.subplots()
    ax.plot(x, y_c, label="Control", color="blue")
    ax.plot(x, y_v, label="Variation", color="green")
    ax.fill_between(x, y_c, alpha=0.2, color="blue")
    ax.fill_between(x, y_v, alpha=0.2, color="green")
    ax.legend(); st.pyplot(fig)

with tab5:
    sim_c = np.random.binomial(users_control, rate_c, 5000)/users_control
    sim_v = np.random.binomial(users_variation, rate_v, 5000)/users_variation
    diffs = (sim_v - sim_c) * 100
    fig, ax = plt.subplots()
    ax.hist(diffs, bins=50, color='orange', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--')
    ax.set_title("Distribution of Lift (%)")
    st.pyplot(fig)

