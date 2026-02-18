import streamlit as st
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import beta, chi2_contingency
from statsmodels.stats.proportion import proportions_ztest, proportion_confint, proportion_effectsize
from statsmodels.stats.power import NormalIndPower, TTestIndPower
import openai

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Enterprise A/B Test Analyzer", layout="wide", initial_sidebar_state="expanded")
matplotlib.use('Agg') # Prevents crashing in non-GUI environments

# --- 2. ASSETS & ICONS (DEFINED FIRST TO PREVENT NAME ERRORS) ---
ICON_SETTINGS = """<svg viewBox="0 0 1024 1024" style="width: 1.2em; height: 1.2em; vertical-align: middle; margin-right: 8px;" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M512 512m-10 0a10 10 0 1 0 20 0 10 10 0 1 0-20 0Z" fill="#E73B37"></path><path d="M512 306.8c27.7 0 54.6 5.4 79.8 16.1 24.4 10.3 46.4 25.1 65.2 44s33.6 40.8 44 65.2c10.7 25.3 16.1 52.1 16.1 79.8 0 27.7-5.4 54.6-16.1 79.8-10.3 24.4-25.1 46.4-44 65.2-18.8 18.8-40.8 33.6-65.2 44-25.3 10.7-52.1 16.1-79.8 16.1-27.7 0-54.6-5.4-79.8-16.1-24.4-10.3-46.4-25.1-65.2-44-18.8-18.8-33.6-40.8-44-65.2-10.7-25.3-16.1-52.1-16.1-79.8 0-27.7 5.4-54.6 16.1-79.8 10.3-24.4 25.1-46.4 44-65.2s40.8-33.6 65.2-44c25.2-10.6 52.1-16.1 79.8-16.1m0-22c-125.4 0-227.1 101.7-227.1 227.1S386.6 739.1 512 739.1c125.4 0 227.1-101.7 227.1-227.1S637.4 284.8 512 284.8z" fill="#ffffff"></path><path d="M512 618.7c-58.9 0-106.8-47.9-106.8-106.8S453.1 405.1 512 405.1 618.8 453 618.8 511.9 570.9 618.7 512 618.7z m0-193.5c-47.9 0-86.8 38.9-86.8 86.8s38.9 86.8 86.8 86.8 86.8-38.9 86.8-86.8-38.9-86.8-86.8-86.8z" fill="#E73B37"></path><path d="M544.2 107.3l34.1 92.3 7.4 19.9 20.2 6.6c10.3 3.4 32.1 12.9 43.4 18.1l18.7 8.6 18.6-8.9 87.9-41.8 46.4 46.5-41.2 89.4-8.9 19.3 9.6 19c6.8 13.4 12.6 27.5 17.4 41.9l6.7 20.5 20.3 7.2 91.7 32.6v65.7l-92.3 34.1-19.9 7.4-6.6 20.2c-4.7 14.4-10.6 28.4-17.4 41.9l-9.8 19.3 9.3 19.5 41.8 87.9-46.5 46.5-89.1-41.3-19.3-8.9-19 9.6c-13.4 6.8-27.5 12.6-41.9 17.4l-20.5 6.7-7.2 20.3-32.6 91.7h-65.7l-34.1-92.3-7.4-19.9-20.2-6.6c-10.3-3.4-32.1-12.9-43.4-18.1L356 771l-18.6 8.9-87.9 41.8-46.4-46.5 41.2-89.3 8.9-19.3-9.6-19c-6.8-13.4-12.6-27.5-17.4-41.9l-6.7-20.5-20.3-7.2-91.7-32.6v-65.7l92.3-34.1 19.9-7.4 6.6-20.2c3.4-10.3 12.9-32.1 18.1-43.4l8.6-18.7-8.9-18.6-41.8-87.9 46.4-46.4 89.3 41.2 19.3 8.9 19-9.6c13.4-6.8 27.5-12.6 41.9-17.4l20.5-6.7 7.2-20.3 32.6-91.7h65.7m30.7-44.1H447.4l-43 121c-16.6 5.5-32.7 12.1-48.1 19.9l-117.2-54-90.1 90.1 55.2 116s-14.5 31.4-19.9 48.1l-121 44.7v127.4l121 43c5.5 16.6 12.1 32.6 19.9 48l-54 117.2 90.1 90.1 116-55.2s31.4 14.5 48.1 19.9l44.7 121h127.4l43-121c16.6-5.5 32.6-12.1 48-19.9l117.2 54 90.1-90.1-55.2-116c7.8-15.4 14.5-31.4 19.9-48l121-44.7V447.4l-121-43c-5.5-16.6-12.1-32.6-19.9-48l54-117.2-90.1-90.1-115.9 55.2s-31.5-14.5-48.1-19.9L574.9 63.3z" fill="#ffffff"></path></svg>"""
ICON_UPLOAD = """<svg viewBox="0 0 1024 1024" style="width: 1.2em; height: 1.2em; vertical-align: middle; margin-right: 8px;" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M220.5 245.4c-32.8 32.8-55.1 73.2-65.2 117.3h16.5c18.8-75.3 75.1-135.9 148-160.7v-16.9c-37.1 11.6-71 32-99.3 60.3z" fill="#E73B37"></path><path d="M959.9 540.8c0 113.6-92.1 205.8-205.7 205.9H590.9v-44h163.3c43.2 0 83.8-16.9 114.3-47.4 30.6-30.6 47.4-71.2 47.4-114.5 0-43.2-16.8-83.9-47.4-114.4S797.2 379 754 379c-11.5 0-22.8 1.2-33.8 3.5-15 3.2-29.4 8.4-42.8 15.7-1-15.4-3.3-30.7-6.8-45.6v-0.1c-3.6-15.6-8.6-30.8-14.9-45.7-14.4-33.9-34.9-64.4-61.1-90.6-26.2-26.2-56.6-46.7-90.6-61.1-35.1-14.8-72.4-22.4-110.9-22.4s-75.8 7.5-110.9 22.4c-33.9 14.3-64.4 34.9-90.6 61.1-26.2 26.2-46.7 56.7-61.1 90.6-14.9 35.1-22.4 72.4-22.4 110.9s7.5 75.8 22.4 110.9c14.3 33.9 34.9 64.4 61.1 90.6 26.2 26.2 56.7 46.7 90.6 61.1 35.1 14.8 72.4 22.4 110.9 22.4h39.7v44h-41C210.7 746 64.1 599 64.1 417.7c0-181.7 147.3-329 329-329 154.6 0 284.3 106.6 319.5 250.3v0.1c13.4-2.7 27.2-4.2 41.4-4.2 113.7 0.1 205.9 92.2 205.9 205.9z" fill="#ffffff"></path><path d="M692.9 636.1h-22.6L519.8 485.6v449.6h-16V485.8L353.4 636.1h-22.6l181-181z" fill="#E73B37"></path></svg>"""
ICON_TROPHY = """<svg viewBox="0 0 1024 1024" style="width: 1.2em; height: 1.2em; vertical-align: middle; margin-right: 8px;" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M828.5 180.1h-9.9v-54.7h23.5v-44H182v44h23v54.7h-9.5C123.2 180.1 64 239.2 64 311.5v0.1c0 72.3 59.2 131.5 131.5 131.5h9.6c0 1.3 0.1 2.5 0.1 3.7 0.5 17.7 2.7 35.4 6.2 52.5 17.8 85.7 71.8 160 148.3 204 4.8 2.8 9.8 5.4 14.7 7.9 15.3 7.7 31.2 14.1 47.4 19.2 3.4 1 6.8 2 10.2 2.9v165.2H250.4v44h511.9v-44H591.9V733.4c3.7-1 7.3-2.1 10.9-3.2 16.2-5.1 32.2-11.6 47.4-19.4 5-2.5 10-5.3 14.8-8.1 75.6-43.9 129.2-117.8 147-202.7 3.6-17.2 5.8-34.9 6.3-52.4 0.1-1.5 0.1-3 0.1-4.5h10c72.3 0 131.5-59.2 131.5-131.5v-0.1c0.1-72.3-59.1-131.4-131.4-131.4zM205 399.2h-9.5c-23.2 0-45.1-9.1-61.7-25.7s-25.7-38.5-25.7-61.7v-0.1c0-23.2 9.1-45.2 25.7-61.7 16.6-16.6 38.5-25.7 61.7-25.7h9.5v174.9z m370.9 499.4h-128V737.3c20.9 4.5 42.3 6.8 63.9 6.8 21.7 0 43.1-2.3 64.1-6.8v161.3z m198.7-461.4c0 2.9 0 5.9-0.2 8.9-0.5 15-2.3 30.1-5.4 44.9-15.3 72.7-61.2 136-126.1 173.7-4.1 2.4-8.4 4.7-12.7 6.9-13 6.6-26.7 12.2-40.6 16.6-25.2 7.9-51.4 11.9-77.9 11.9-26.2 0-52.2-3.9-77.1-11.6-13.9-4.3-27.5-9.8-40.6-16.4-4.2-2.1-8.5-4.4-12.6-6.8-65.4-37.8-111.7-101.5-126.9-174.8-3.1-14.7-4.9-29.8-5.3-45-0.1-2.7-0.1-5.5-0.1-8.2v-312h525.6v311.9zM916 311.7c0 23.2-9.1 45.2-25.7 61.7-16.6 16.6-38.5 25.7-61.7 25.7h-9.9v-175h9.9c23.2 0 45.1 9.1 61.7 25.7s25.7 38.5 25.7 61.7v0.2z" fill="#ffffff"></path><path d="M317.428 274.917l70.145-70.144 14.142 14.142-70.145 70.144zM316.055 351.98L456.13 211.904l14.142 14.142-140.076 140.076zM555.4 659.6l-4.8-19.4c0.3-0.1 26.5-6.8 55.4-23.5 37.8-21.9 62-49.7 72-82.7l19.1 5.8c-11.4 37.6-39.6 70.3-81.6 94.5-31.2 18-58.9 25-60.1 25.3z" fill="#E73B37"></path></svg>"""
ICON_CALENDAR = """<svg viewBox="0 0 1024 1024" style="width: 1.2em; height: 1.2em; vertical-align: middle; margin-right: 8px;" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M716 190.9v-67.8h-44v67.8H352v-67.8h-44v67.8H92v710h840v-710H716z m-580 44h172v69.2h44v-69.2h320v69.2h44v-69.2h172v151.3H136V234.9z m752 622H136V402.2h752v454.7z" fill="#ffffff"></path><path d="M319 565.7m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"></path><path d="M510 565.7m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"></path><path d="M701.1 565.7m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"></path><path d="M319 693.4m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"></path><path d="M510 693.4m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"></path><path d="M701.1 693.4m-33 0a33 33 0 1 0 66 0 33 33 0 1 0-66 0Z" fill="#E73B37"></path></svg>"""
ICON_PIE = """<svg viewBox="0 0 1024 1024" style="width: 1.2em; height: 1.2em; vertical-align: middle; margin-right: 8px;" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M429.9 186.7v406.4h407.5c-4 34.1-12.8 67.3-26.2 99.1-18.4 43.6-44.8 82.7-78.5 116.3-33.6 33.6-72.8 60-116.4 78.4-45.1 19.1-93 28.7-142.5 28.7-49.4 0-97.4-9.7-142.5-28.7-43.6-18.4-82.7-44.8-116.4-78.4-33.6-33.6-60-72.7-78.4-116.3-19.1-45.1-28.7-93-28.7-142.4s9.7-97.3 28.7-142.4c18.4-43.6 44.8-82.7 78.4-116.3 33.6-33.6 72.8-60 116.4-78.4 31.7-13.2 64.7-21.9 98.6-26m44-46.6c-226.4 0-410 183.5-410 409.8s183.6 409.8 410 409.8 410-183.5 410-409.8v-0.8h-410v-409z" fill="#ffffff"></path><path d="M566.1 80.5c43.7 1.7 86.4 10.6 127 26.4 44 17.1 84.2 41.8 119.6 73.5 71.7 64.1 117.4 151.7 128.7 246.7 1.2 9.9 2 20 2.4 30.2H566.1V80.5m-16-16.3v409h410c0-16.3-1-32.3-2.9-48.1C933.1 221.9 760 64.2 550.1 64.2zM264.7 770.4c-23.1-23.1-42.3-49.1-57.3-77.7l-14.7 6.5c35.7 68.2 94 122.7 165 153.5l4.3-15.6c-36.3-16-69.1-38.4-97.3-66.7z" fill="#E73B37"></path></svg>"""
ICON_BAR_CHART = """<svg viewBox="0 0 1024 1024" style="width: 1.2em; height: 1.2em; vertical-align: middle; margin-right: 8px;" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M928.1 881v44H95.9V99h44v782z" fill="#ffffff"></path><path d="M352 435.7v403.4H204V435.7h148m22-22H182v447.4h192V413.7zM608 307.9v531.2H460V307.9h148m22-22H438v575.2h192V285.9z" fill="#ffffff"></path><path d="M866.1 177.3v663.9H714V177.3h152.1m20-20H694v703.9h192V157.3h0.1z" fill="#E73B37"></path></svg>"""
ICON_BRAIN = """<svg viewBox="0 0 1024 1024" style="width: 1.2em; height: 1.2em; vertical-align: middle; margin-right: 8px;" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M512 301.2m-10 0a10 10 0 1 0 20 0 10 10 0 1 0-20 0Z" fill="#E73B37"></path><path d="M400.3 744.5c2.1-0.7 4.1-1.4 6.2-2-2 0.6-4.1 1.3-6.2 2z m0 0c2.1-0.7 4.1-1.4 6.2-2-2 0.6-4.1 1.3-6.2 2z" fill="#ffffff"></path><path d="M511.8 256.6c24.4 0 44.2 19.8 44.2 44.2S536.2 345 511.8 345s-44.2-19.8-44.2-44.2 19.9-44.2 44.2-44.2m0-20c-35.5 0-64.2 28.7-64.2 64.2s28.7 64.2 64.2 64.2 64.2-28.7 64.2-64.2-28.7-64.2-64.2-64.2z" fill="#E73B37"></path><path d="M730.7 529.5c0.4-8.7 0.6-17.4 0.6-26.2 0-179.6-86.1-339.1-219.3-439.5-133.1 100.4-219.2 259.9-219.2 439.5 0 8.8 0.2 17.5 0.6 26.1-56 56-90.6 133.3-90.6 218.7 0 61.7 18 119.1 49.1 167.3 30.3-49.8 74.7-90.1 127.7-115.3 39-18.6 82.7-29 128.8-29 48.3 0 93.9 11.4 134.3 31.7 52.5 26.3 96.3 67.7 125.6 118.4 33.4-49.4 52.9-108.9 52.9-173.1 0-85.4-34.6-162.6-90.5-218.6z m-90.6 449.2c-9.1-27-13.7-55.5-13.7-84.4 0-35.8 7-70.6 20.8-103.2 8.4-19.8 19-38.4 31.9-55.5 9.7 61.5 29.5 119.7 57.8 172.6-36.4 17.8-69 41.6-96.8 70.5z m364.2-85.3c-0.7-0.3-1.5-0.5-2.2-0.8-0.4-0.2-0.9-0.3-1.3-0.5-0.6-0.2-1.3-0.5-1.9-0.7-0.8-0.3-1.5-0.5-2.3-0.8-0.8-0.3-1.5-0.5-2.3-0.7l-0.9-0.3c-1-0.3-2.1-0.7-3.1-1-1.2-0.4-2.4-0.7-3.5-1.1l-3-0.9c-0.2-0.1-0.4-0.1-0.7-0.2-1.1-0.3-2.3-0.7-3.4-1-1.2-0.3-2.4-0.6-3.5-0.9l-3.6-0.9-3.6-0.9c-1-0.3-2.1-0.5-3.1-0.7-1.2-0.3-2.4-0.5-3.6-0.8-1.3-0.3-2.5-0.6-3.8-0.8h-0.3c-0.9-0.2-1.9-0.4-2.8-0.6-0.4-0.1-0.7-0.1-1.1-0.2-1.1-0.2-2.2-0.4-3.4-0.6-1.2-0.2-2.4-0.4-3.6-0.7l-5.4-0.9c-0.9-0.1-1.9-0.3-2.8-0.4-0.8-0.1-1.6-0.3-2.5-0.4-2.6-0.4-5.1-0.7-7.7-1-1.2-0.1-2.3-0.3-3.5-0.4h-0.4c-0.9-0.1-1.8-0.2-2.8-0.3-1.1-0.1-2.1-0.2-3.2-0.3-1.7-0.2-3.4-0.3-5.1-0.4-0.8-0.1-1.5-0.1-2.3-0.2-0.9-0.1-1.9-0.1-2.8-0.2-0.4 0-0.8 0-1.2-0.1-1.1-0.1-2.1-0.1-3.2-0.2-0.5 0-1-0.1-1.5-0.1-1.3-0.1-2.6-0.1-3.9-0.1-0.8 0-1.5-0.1-2.3-0.1-1.2 0-2.4 0-3.5-0.1h-13.9c-2.3 0-4.6 0.1-6.9 0.2-0.9 0-1.9 0.1-2.8 0.1-0.8 0-1.5 0.1-2.3 0.1-1.4 0.1-2.8 0.2-4.1 0.3-1.4 0.1-2.7 0.2-4.1 0.3-1.4 0.1-2.7 0.2-4.1 0.4-0.6 0-1.2 0.1-1.8 0.2l-7.8 0.9c-1.1 0.1-2.1 0.3-3.2 0.4-1 0.1-2.1 0.3-3.1 0.4-3.2 0.5-6.4 0.9-9.5 1.5-0.7 0.1-1.4 0.2-2.1 0.4-0.9 0.1-1.7 0.3-2.6 0.5-1.1 0.2-2.3 0.4-3.4 0.6-0.9 0.2-1.7 0.3-2.6 0.5-0.4 0.1-0.8 0.1-1.1 0.2-0.7 0.1-1.4 0.3-2.1 0.4-1.2 0.3-2.4 0.5-3.6 0.8-1.2 0.3-2.4 0.5-3.6 0.8-0.2 0-0.4 0.1-0.6 0.1-0.5 0.1-1 0.2-1.5 0.4-1.1 0.3-2.3 0.6-3.5 0.9-1.3 0.3-2.5 0.6-3.8 1-0.4 0.1-0.9 0.2-1.4 0.4-1.3 0.4-2.7 0.7-4 1.1-1.5 0.4-3 0.9-4.6 1.3-1 0.3-2.1 0.6-3.1 1-2.1 0.6-4.1 1.3-6.2 2-0.7 0.2-1.4 0.5-2.1 0.7-15-27.5-27.4-56.4-37-86.2-11.7-36.1-19.2-73.6-22.5-111.6-0.6-6.7-1-13.3-1.3-20-0.1-1.2-0.1-2.4-0.1-3.6-0.1-1.2-0.1-2.4-0.1-3.6 0-1.2-0.1-2.4-0.1-3.6 0-1.2-0.1-2.4-0.1-3.7 18.8-14 39.2-25.8 61-35 36.1-15.3 74.5-23 114.1-23 39.6 0 78 7.8 114.1 23 21.8 9.2 42.2 20.9 61 35v0.1c0 1 0 1.9-0.1 2.9 0 1.4-0.1 2.8-0.1 4.3 0 0.7 0 1.3-0.1 2-0.1 1.8-0.1 3.5-0.2 5.3-0.3 6.7-0.8 13.3-1.3 20-3.3 38.5-11 76.5-23 113-9.7 30.3-22.3 59.4-37.6 87.1z m136.8 90.9a342.27 342.27 0 0 0-96.3-73.2c29.1-53.7 49.5-112.8 59.4-175.5 12.8 17.1 23.4 35.6 31.8 55.5 13.8 32.7 20.8 67.4 20.8 103.2 0 31-5.3 61.3-15.7 90z" fill="#ffffff"></path><path d="M512 819.3c8.7 0 24.7 22.9 24.7 60.4s-16 60.4-24.7 60.4-24.7-22.9-24.7-60.4 16-60.4 24.7-60.4m0-20c-24.7 0-44.7 36-44.7 80.4 0 44.4 20 80.4 44.7 80.4s44.7-36 44.7-80.4c0-44.4-20-80.4-44.7-80.4z" fill="#E73B37"></path></svg>"""

# --- 3. HELPER & PLOTTING FUNCTIONS (DEFINED FIRST TO PREVENT NAME ERRORS) ---
def render_header(svg_code, text, level=2):
    st.markdown(f"""<div style="display: flex; align-items: center; margin-bottom: 10px;">{svg_code}<h{level} style="margin: 0; padding: 0;">{text}</h{level}></div>""", unsafe_allow_html=True)

def calculate_uplift(ctrl, var):
    if ctrl == 0: return 0.0
    return ((var - ctrl) / ctrl) * 100

def perform_srm_test(observed_users, expected_split=(0.5, 0.5)):
    total_users = sum(observed_users)
    expected_users = [total_users * p for p in expected_split]
    chi2_stat, p_value = chi2_contingency([observed_users, expected_users])[:2]
    return chi2_stat, p_value

def calculate_bayesian_risk(alpha_c, beta_c, alpha_v, beta_v, num_samples=50000):
    samples_c = np.random.beta(alpha_c, beta_c, num_samples)
    samples_v = np.random.beta(alpha_v, beta_v, num_samples)
    prob_v_wins = np.mean(samples_v > samples_c)
    loss_v = np.mean(np.maximum(samples_c - samples_v, 0))
    loss_c = np.mean(np.maximum(samples_v - samples_c, 0))
    return prob_v_wins, loss_v, loss_c

def check_simpsons_paradox(seg1, seg2):
    cr_c1 = seg1['conv_c'] / seg1['users_c'] if seg1['users_c'] > 0 else 0
    cr_v1 = seg1['conv_v'] / seg1['users_v'] if seg1['users_v'] > 0 else 0
    uplift_1 = calculate_uplift(cr_c1, cr_v1)
    
    cr_c2 = seg2['conv_c'] / seg2['users_c'] if seg2['users_c'] > 0 else 0
    cr_v2 = seg2['conv_v'] / seg2['users_v'] if seg2['users_v'] > 0 else 0
    uplift_2 = calculate_uplift(cr_c2, cr_v2)
    
    agg_users_c = seg1['users_c'] + seg2['users_c']
    agg_conv_c = seg1['conv_c'] + seg2['conv_c']
    agg_users_v = seg1['users_v'] + seg2['users_v']
    agg_conv_v = seg1['conv_v'] + seg2['conv_v']
    
    cr_c_agg = agg_conv_c / agg_users_c if agg_users_c > 0 else 0
    cr_v_agg = agg_conv_v / agg_users_v if agg_users_v > 0 else 0
    uplift_agg = calculate_uplift(cr_c_agg, cr_v_agg)
    
    paradox = False
    if (uplift_1 > 0 and uplift_2 > 0 and uplift_agg < 0): paradox = True
    elif (uplift_1 < 0 and uplift_2 < 0 and uplift_agg > 0): paradox = True
        
    return paradox, uplift_1, uplift_2, uplift_agg

def get_ai_analysis(api_key, hypothesis, metrics_dict, provider="OpenAI", conf_level="95%"):
    if not api_key: return "Please enter a valid API Key to generate this analysis."
    
    if provider == "DeepSeek":
        base_url = "https://api.deepseek.com"
        model_name = "deepseek-reasoner"
    else:
        base_url = "https://api.openai.com/v1"
        model_name = "gpt-4o"

    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        prompt = f"""
        Analyze this A/B test results.
        Configuration: Confidence Level: {conf_level}
        Hypothesis: "{hypothesis}"
        Data: {metrics_dict}
        Task: Provide Executive Summary, Trade-off Analysis, Risk Assessment, and Recommendation.
        Format: Markdown.
        """
        response = client.chat.completions.create(
            model=model_name, 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to AI: {str(e)}"

# --- NEW: COMPREHENSIVE AUTO-ANALYST (Rule-Based NLP) ---
def generate_comprehensive_report(hypothesis, metrics, alpha_val):
    report = []
    
    # 1. Executive Summary
    sig = metrics['p_cr'] <= alpha_val
    uplift = metrics['uplift_cr']
    
    if sig and uplift > 0:
        headline = "WINNER: Significant Positive Result"
        summary = f"The variation is statistically superior with {100-metrics['alpha']*100:.0f}% confidence."
    elif sig and uplift < 0:
        headline = "LOSER: Significant Negative Result"
        summary = "The variation is performing significantly worse than the control."
    else:
        headline = "INCONCLUSIVE: No Clear Winner"
        summary = f"We cannot reject the null hypothesis (p={metrics['p_cr']:.3f})."

    report.append(f"### {headline}")
    report.append(summary)
    
    # 2. Deep Dive: Primary Metrics
    report.append("#### Deep Dive: Primary Metrics")
    report.append(f"- Conversion Rate: {metrics['uplift_cr']:.2f}% uplift.")
    report.append(f"- Average Order Value: {metrics['uplift_aov']:.2f}% uplift (${metrics['aov_c']:.2f} vs ${metrics['aov_v']:.2f}).")
    if metrics['uplift_rpv'] > 0:
        report.append(f"- Revenue Per Visitor: POSITIVE. Each visitor is worth ${metrics['rpv_v'] - metrics['rpv_c']:.2f} more.")
    else:
        report.append(f"- Revenue Per Visitor: NEGATIVE. Each visitor is worth ${metrics['rpv_c'] - metrics['rpv_v']:.2f} less.")

    # 3. Deep Dive: Secondary Metrics
    report.append("#### Deep Dive: Product Velocity")
    if metrics['uplift_apo'] > 0:
        report.append(f"- Basket Building: POSITIVE. Users are adding {metrics['uplift_apo']:.2f}% more items per order.")
    else:
        report.append(f"- Basket Building: NEGATIVE. Users are adding {abs(metrics['uplift_apo']):.2f}% fewer items per order.")

    # 4. Statistical Robustness
    report.append("#### Statistical Robustness")
    
    if metrics['p_cr'] < metrics['adjusted_alpha']:
        report.append(f"- Sequential Testing: ROBUST. Result remains significant after peeking penalty (Adj Alpha: {metrics['adjusted_alpha']:.5f}).")
    else:
        report.append("- Sequential Testing: CAUTION. Frequent peeking increases false positive risk.")

    report.append(f"- Bayesian Probability: {metrics['prob_v_wins']:.1f}% chance variation is best.")
    if metrics['loss_v'] < 0.01:
        report.append("- Risk: LOW. The expected loss from switching is negligible.")
    else:
        report.append(f"- Risk: MODERATE/HIGH. Potential loss: {metrics['loss_v']:.4f}.")

    width = metrics['ci_high'] - metrics['ci_low']
    if width < 5.0:
        report.append(f"- Bootstrap Confidence: PRECISE. Narrow interval ({width:.2f}%) indicates stable behavior.")
    else:
        report.append(f"- Bootstrap Confidence: VOLATILE. Wide interval ({width:.2f}%) indicates high variance.")

    # 5. Strategic Conclusion
    report.append("#### Strategic Conclusion")
    if metrics['uplift_cr'] > 0 and metrics['uplift_aov'] > 0:
        report.append("Growth Engine: Efficient win driving both volume and value.")
    elif metrics['uplift_cr'] > 0 and metrics['uplift_aov'] < 0:
        report.append("Discount Effect: Driving volume at expense of value.")
    elif metrics['uplift_cr'] < 0 and metrics['uplift_aov'] > 0:
        report.append("Premium Shift: Higher value customers, but losing volume.")
    else:
        report.append("Negative Friction: Both conversion and value are down.")

    return "\n\n".join(report)

def plot_strategic_matrix(cr_c, aov_c, cr_v, aov_v):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cr_c, aov_c, color='blue', s=200, label='Control', zorder=5)
    ax.scatter(cr_v, aov_v, color='green', s=200, label='Variation', zorder=5)
    ax.annotate("", xy=(cr_v, aov_v), xytext=(cr_c, aov_c), arrowprops=dict(arrowstyle="->", color='gray', lw=1.5, ls='--'))
    ax.axvline(cr_c, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(aov_c, color='gray', linestyle=':', alpha=0.5)
    ax.set_title("Strategic Matrix: Volume (CR) vs Value (AOV)")
    ax.set_xlabel("Conversion Rate (%)")
    ax.set_ylabel("Average Order Value ($)")
    ax.legend()
    st.pyplot(fig)

def plot_metric_comparison(name, val_c, val_v, unit=""):
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = ['Control', 'Variation']
    values = [val_c, val_v]
    colors = ['#1f77b4', '#2ca02c'] if val_v >= val_c else ['#1f77b4', '#d62728']
    ax.bar(groups, values, color=colors, alpha=0.8)
    ax.set_title(f'{name} Comparison')
    ax.set_ylabel(name)
    ax.set_ylim(0, max(values) * 1.15)
    for i, v in enumerate(values):
        ax.text(i, v + (max(values)*0.01), f"{unit}{v:.2f}", ha='center', va='bottom', fontweight='bold')
    st.pyplot(fig)

def plot_bayesian_pdfs(alpha_c, beta_c, alpha_v, beta_v):
    x_min = min(beta(alpha_c, beta_c).ppf(0.001), beta(alpha_v, beta_v).ppf(0.001))
    x_max = max(beta(alpha_c, beta_c).ppf(0.999), beta(alpha_v, beta_v).ppf(0.999))
    x = np.linspace(x_min, x_max, 1000)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, beta(alpha_c, beta_c).pdf(x), label='Control', color='blue')
    ax.fill_between(x, beta(alpha_c, beta_c).pdf(x), 0, alpha=0.3, color='blue')
    ax.plot(x, beta(alpha_v, beta_v).pdf(x), label='Variation', color='green')
    ax.fill_between(x, beta(alpha_v, beta_v).pdf(x), 0, alpha=0.3, color='green')
    ax.set_title('Bayesian Probability Density')
    ax.legend()
    st.pyplot(fig)

def run_bootstrap_and_plot(users_c, conv_c, users_v, conv_v, n_bootstrap=10000, alpha_val=0.05):
    sim_c = np.random.binomial(users_c, conv_c/users_c, n_bootstrap) / users_c
    sim_v = np.random.binomial(users_v, conv_v/users_v, n_bootstrap) / users_v
    diffs = sim_v - sim_c
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(diffs, bins=50, color='orange', edgecolor='black', alpha=0.7)
    lower_p = (alpha_val / 2) * 100
    upper_p = (1 - (alpha_val / 2)) * 100
    lower_ci = np.percentile(diffs, lower_p)
    upper_ci = np.percentile(diffs, upper_p)
    ax.axvline(lower_ci, color='red', linestyle='--', label=f'CI Lower')
    ax.axvline(upper_ci, color='red', linestyle='--', label=f'CI Upper')
    ax.set_title('Bootstrap Distribution')
    ax.legend()
    st.pyplot(fig)
    return sim_c * 100, sim_v * 100, lower_ci * 100, upper_ci * 100

def plot_box_plot_analysis(sim_samples_c, sim_samples_v):
    data = [sim_samples_c, sim_samples_v]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, boxprops=dict(facecolor="skyblue")) 
    ax.set_xticklabels(['Control Group', 'Variation Group'])
    ax.set_ylabel('Conversion Rate (%)')
    ax.set_title('Box Plot Analysis')
    st.pyplot(fig)

# --- 4. STATE INITIALIZATION (FIXES WIDGET WARNINGS) ---
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

# --- 5. SIDEBAR UI (COHESIVE SETTINGS) ---
st.sidebar.markdown(f"""<div style="display: flex; align-items: center; margin-bottom: 10px;">{ICON_SETTINGS}<h2 style="display: inline; font-size: 1.5rem; margin-left: 5px;">Settings</h2></div>""", unsafe_allow_html=True)

# Sub-Section: Save & Load
st.sidebar.markdown(f"""<div style="display: flex; align-items: center; margin-top: 10px; margin-bottom: 5px;">{ICON_UPLOAD}<span style="font-size: 1rem; font-weight: 600; margin-left: 5px;">Save & Load Analysis</span></div>""", unsafe_allow_html=True)
with st.sidebar.expander("Manage Files", expanded=False):
    st.info("Workflow Continuity: Use this to capture a snapshot of your experiment. Download data on Day 7, then re-upload on Day 14 to see how Probability and Risk evolve without re-typing baseline settings.")
    
    current_state = {k: st.session_state[k] for k in ["users_c", "conv_c", "rev_c", "prod_c", "users_v", "conv_v", "rev_v", "prod_v", "days", "conf_level", "s1_uc", "s1_cc", "s1_uv", "s1_cv", "s2_uc", "s2_cc", "s2_uv", "s2_cv"]}
    st.download_button(label="Download Inputs (.json)", data=json.dumps(current_state, indent=2), file_name="experiment_data.json", mime="application/json")
    uploaded_file = st.file_uploader("Load Analysis", type=["json"])
    if uploaded_file:
        try:
            data = json.load(uploaded_file)
            for k, v in data.items(): st.session_state[k] = v
            st.success("Loaded!")
            st.rerun()
        except: st.error("Invalid File")

# Sub-Section: Confidence
confidence_level = st.sidebar.selectbox(
    "Confidence Level",
    ["95%", "90%", "99%"],
    index=0,
    key="conf_level",
    help="Select statistical significance threshold ~ ⚡ 90% - Fast (Marketing copy, colors, low-risk UI tweaks.) | ⚖️ 95% - Balanced (Standard features, flows, and layouts.) | 🐢 99% - Slow (Pricing changes, checkout logic, backend algorithms.)"
)
alpha = {"90%": 0.10, "95%": 0.05, "99%": 0.01}[confidence_level]

st.sidebar.markdown("---")

# Planning
st.sidebar.markdown(f"""<div style="display: flex; align-items: center; margin-bottom: 10px;">{ICON_CALENDAR}<h2 style="display: inline; font-size: 1.5rem; margin-left: 5px;">Experiment Planning</h2></div>""", unsafe_allow_html=True)
with st.sidebar.expander("Sample Size Calculator", expanded=False):
    st.caption(f"Plan required traffic using {confidence_level} confidence.")
    
    plan_traffic_28d = st.number_input("Traffic (Last 28 Days)", step=1000, key="p_traffic")
    plan_base_cr = st.number_input("Baseline Conversion Rate (%)", step=0.1, key="p_base_cr")
    plan_base_aov = st.number_input("Baseline AOV ($)", step=1.0, key="p_base_aov")
    plan_mde = st.number_input("Min Detectable Effect (%)", step=0.5, key="p_mde")
    
    volatility = st.selectbox(
        "Revenue Variance", 
        ["Low (Subscription)", "Medium (Standard E-com)", "High (Whales/B2B)"],
        index=1,
        key="p_vol"
    )
    
    sd_multiplier = 1.0 if "Low" in volatility else (2.0 if "Medium" in volatility else 3.0)

    if st.button("Calculate Duration"):
        daily_traffic = plan_traffic_28d / 28
        p1 = plan_base_cr / 100
        p2 = p1 * (1 + plan_mde/100)
        effect_size_cr = proportion_effectsize(p1, p2)
        n_cr = NormalIndPower().solve_power(effect_size=effect_size_cr, alpha=alpha, power=0.8, ratio=1)
        
        mean_1 = plan_base_aov
        mean_2 = plan_base_aov * (1 + plan_mde/100)
        est_sd = plan_base_aov * sd_multiplier
        effect_size_rpv = (mean_2 - mean_1) / est_sd
        n_rpv = TTestIndPower().solve_power(effect_size=effect_size_rpv, alpha=alpha, power=0.8, ratio=1)
        
        n_rpv_visitors = n_rpv / (plan_base_cr / 100)
        days_cr = (n_cr * 2) / daily_traffic 
        days_rpv = (n_rpv_visitors * 2) / daily_traffic

        st.markdown("---")
        st.write(f"**Daily Traffic:** {int(daily_traffic):,} users")
        st.info(f"**Conversion Rate:**\nNeed {int(days_cr)} days\n({int(n_cr):,} users/group)")
        
        if days_rpv > 60:
            st.error(f"**Revenue (RPV):**\nNeed {int(days_rpv)} days\n({int(n_rpv_visitors):,} users/group)")
        else:
            st.warning(f"**Revenue (RPV):**\nNeed {int(days_rpv)} days\n({int(n_rpv_visitors):,} users/group)")

st.sidebar.markdown("---")

# Enter Results (No default values to avoid warnings)
st.sidebar.markdown(f"""<div style="display: flex; align-items: center; margin-bottom: 10px;">{ICON_TROPHY}<h2 style="display: inline; font-size: 1.5rem; margin-left: 5px;">Enter Results</h2></div>""", unsafe_allow_html=True)
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

# --- 6. CALCULATIONS & DASHBOARD (RUNS AFTER ALL DEFINITIONS) ---
# Perform all calculations BEFORE rendering tabs to avoid NameError
rate_c = conv_control / users_control if users_control > 0 else 0
rate_v = conv_variation / users_variation if users_variation > 0 else 0
up_cr = calculate_uplift(rate_c, rate_v)

aov_c = rev_control / conv_control if conv_control > 0 else 0
aov_v = rev_variation / conv_variation if conv_variation > 0 else 0
up_aov = calculate_uplift(aov_c, aov_v)

rpv_c = rev_control / users_control if users_control > 0 else 0
rpv_v = rev_variation / users_variation if users_variation > 0 else 0
up_rpv = calculate_uplift(rpv_c, rpv_v)

apo_c = prod_control / conv_control if conv_control > 0 else 0
apo_v = prod_variation / conv_variation if conv_variation > 0 else 0
up_apo = calculate_uplift(apo_c, apo_v)

apu_c = prod_control / users_control if users_control > 0 else 0
apu_v = prod_variation / users_variation if users_variation > 0 else 0
up_apu = calculate_uplift(apu_c, apu_v)

z_stat, p_val = proportions_ztest([conv_control, conv_variation], [users_control, users_variation])
srm_chi, p_srm = perform_srm_test([users_control, users_variation])

# Dashboard UI
st.title("A/B Test Analyzer v1.7.4")
render_header(ICON_BAR_CHART, "Results Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Conversion Rate", f"{rate_v*100:.2f}%", f"{up_cr:+.2f}%")
c2.metric("RPV", f"${rpv_v:.2f}", f"{up_rpv:+.2f}%")
c3.metric("AOV", f"${aov_v:.2f}", f"{up_aov:+.2f}%")
if p_val <= alpha: c4.success(f"CR Sig: YES (p={p_val:.4f})")
else: c4.info(f"CR Sig: NO (p={p_val:.4f})")

# Product Velocity (UPDATED LAYOUT: 4 Columns)
st.subheader("2. Product Velocity")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Products / Order", f"{apo_v:.2f}", f"{up_apo:+.2f}%")
c2.metric("Avg Products / User", f"{apu_v:.2f}", f"{up_apu:+.2f}%")
# c3 is empty to align c4 with the Sig Box above
c4.caption("Higher 'Products / Order' means users are building bigger baskets.")

st.markdown("---")

# Executive Interpretation (RESTORED)
st.subheader("3. Executive Interpretation")
if up_cr > 0 and up_aov < 0: st.warning("Trade-off Detected: Variation drives MORE orders, but SMALLER baskets.")
elif up_cr < 0 and up_aov > 0: st.warning("Trade-off Detected: Variation drives FEWER orders, but BIGGER baskets.")
else: st.success("Clean Result: Trends are consistent.")

if up_rpv > 0: st.write(f"**Financial Impact:** Variation generates **${rpv_v - rpv_c:.2f} more** per visitor.")
else: st.write(f"**Financial Impact:** Variation generates **${rpv_c - rpv_v:.2f} less** per visitor.")

# Health Checks (RESTORED)
st.subheader("4. Health Checks")
c1, c2 = st.columns(2)
with c1:
    if p_srm < 0.01: st.error(f"SRM DETECTED (p={p_srm:.4f})")
    else: st.success(f"SRM PASSED (p={p_srm:.4f})")
with c2:
    if days_run < 7: st.error(f"Too Short ({days_run} Days)")
    elif days_run < 14: st.warning(f"Short Duration ({days_run} Days)")
    else: st.success(f"Duration OK ({days_run} Days)")

st.markdown("---")
render_header(ICON_PIE, "Advanced: Simpson's Paradox Detector (Segment Analysis)", level=3)
with st.expander("Expand to check segments", expanded=False):
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("Segment 1")
        s1_uc = st.number_input("Control Users", key="s1_uc")
        s1_cc = st.number_input("Control Conv.", key="s1_cc")
        s1_uv = st.number_input("Var Users", key="s1_uv")
        s1_cv = st.number_input("Var Conv.", key="s1_cv")
    with col_s2:
        st.subheader("Segment 2")
        s2_uc = st.number_input("Control Users", key="s2_uc")
        s2_cc = st.number_input("Control Conv.", key="s2_cc")
        s2_uv = st.number_input("Var Users", key="s2_uv")
        s2_cv = st.number_input("Var Conv.", key="s2_cv")
    
    if st.button("Check for Paradox"):
        seg1 = {'users_c': s1_uc, 'conv_c': s1_cc, 'users_v': s1_uv, 'conv_v': s1_cv}
        seg2 = {'users_c': s2_uc, 'conv_c': s2_cc, 'users_v': s2_uv, 'conv_v': s2_cv}
        is_paradox, up1, up2, up_agg = check_simpsons_paradox(seg1, seg2)
        if is_paradox: st.error("PARADOX DETECTED! Segment trends contradict aggregate results.")
        else: st.success("No Paradox. Trends are consistent.")

st.markdown("---")
render_header(ICON_BRAIN, "Deep Dive Analysis")
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "Smart Analysis", "AI Analysis", "Stopping & Sequential", "Strategic Matrix", 
    "Product Metrics", "Revenue Charts", "CR Comparison", "Bayesian", "Bootstrap", "Box Plot"
])

with tab1:
    st.markdown("### Auto-Analyst Report")
    st.info("Generated instantly using rule-based logic (No API Key required).")
    user_hypothesis = st.text_area("Hypothesis:", placeholder="We believed that...", height=70, key="hyp_smart")
    
    peeks_for_report = 1 
    adj_alpha = alpha * np.log(1 + (np.e - 1) / peeks_for_report)

    if st.button("Generate Report"):
        ci_low, ci_high = proportion_confint(conv_variation, users_variation, alpha=alpha, method='normal')
        diff_ci_low = (ci_low - rate_c) * 100
        diff_ci_high = (ci_high - rate_c) * 100
        prob_v_wins, loss_v, loss_c = calculate_bayesian_risk(conv_control+1, users_control-conv_control+1, conv_variation+1, users_variation-conv_variation+1)

        metrics_payload = {
            "days": days_run, "users_c": users_control, "users_v": users_variation, "p_srm": p_srm,
            "cr_c": rate_c*100, "cr_v": rate_v*100, "uplift_cr": up_cr, "p_cr": p_val,
            "aov_c": aov_c, "aov_v": aov_v, "uplift_aov": up_aov, "rpv_c": rpv_c, "rpv_v": rpv_v, "uplift_rpv": up_rpv,
            "ci_low": diff_ci_low, "ci_high": diff_ci_high, "uplift_apo": up_apo,
            "prob_v_wins": prob_v_wins * 100, "loss_v": loss_v * 100, "loss_c": loss_c * 100, "alpha": alpha,
            "adjusted_alpha": adj_alpha
        }
        smart_result = generate_comprehensive_report(user_hypothesis, metrics_payload, alpha)
        st.markdown("---")
        st.markdown(smart_result)

with tab2:
    st.markdown("### AI-Powered Analysis")
    col_prov, col_key = st.columns(2)
    with col_prov:
        ai_provider = st.selectbox("Select Provider", ["OpenAI (GPT-4o)", "DeepSeek (R1)"])
    with col_key:
        api_key_input = st.text_input("Enter API Key", type="password")
    user_hypothesis_ai = st.text_area("Hypothesis (AI):", placeholder="We believed that...", height=100, key="hyp_ai")
    if st.button("Generate AI Analysis"):
        st.info("Connecting to AI service...")

with tab3:
    st.markdown("### Stopping Rules & Sequential Testing")
    prob_v_wins, loss_v, loss_c = calculate_bayesian_risk(conv_control+1, users_control-conv_control+1, conv_variation+1, users_variation-conv_variation+1)
    c1, c2, c3 = st.columns(3)
    c1.metric("Probability Best", f"{prob_v_wins*100:.1f}%")
    c2.metric("Risk (Switch)", f"{loss_v*100:.5f}%")
    c3.metric("Risk (Stay)", f"{loss_c*100:.5f}%")
    st.markdown("---")
    st.subheader("Sequential Testing Penalty (Peeking)")
    peeks = st.number_input("How many times have you checked?", min_value=1, value=1)
    adjusted_alpha = alpha * np.log(1 + (np.e - 1) / peeks)
    st.write(f"Adjusted Threshold: **{adjusted_alpha:.5f}**")
    if p_val < adjusted_alpha: st.success(f"SIGNIFICANT (p={p_val:.4f})")
    else: st.error(f"NOT SIGNIFICANT (p={p_val:.4f})")

with tab4: plot_strategic_matrix(rate_c*100, aov_c, rate_v*100, aov_v)
with tab5: 
    c1, c2 = st.columns(2)
    with c1: plot_metric_comparison("Avg Products / Order", apo_c, apo_v)
    with c2: plot_metric_comparison("Avg Products / User", apu_c, apu_v)
with tab6:
    c1, c2 = st.columns(2)
    with c1: plot_metric_comparison("Revenue Per Visitor", rpv_c, rpv_v, "$")
    with c2: plot_metric_comparison("Avg Order Value", aov_c, aov_v, "$")
with tab7: plot_metric_comparison("Conversion Rate", rate_c*100, rate_v*100)
with tab8: plot_bayesian_pdfs(conv_control+1, users_control-conv_control+1, conv_variation+1, users_variation-conv_variation+1)
with tab9: 
    samples_c, samples_v, ci_low, ci_high = run_bootstrap_and_plot(users_control, conv_control, users_variation, conv_variation, alpha_val=alpha)
    st.write(f"**{confidence_level} CI:** {ci_low:.2f}% to {ci_high:.2f}%")
with tab10: plot_box_plot_analysis(samples_c, samples_v)
