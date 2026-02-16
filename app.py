import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, chi2_contingency, ttest_ind_from_stats
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.power import NormalIndPower

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="A/B Test Analyzer", page_icon="📊", layout="wide")
st.title("📊 Professional A/B Test Analyzer (Full Suite)")

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("Experiment Data")

# Control
st.sidebar.subheader("Control Group")
users_control = st.sidebar.number_input("Control Users", min_value=1, value=5000)
conv_control = st.sidebar.number_input("Control Conversions", min_value=0, value=500)
rev_control = st.sidebar.number_input("Control Revenue ($)", min_value=0.0, value=25000.0)
prod_control = st.sidebar.number_input("Control Products Sold", min_value=0, value=750)

st.sidebar.markdown("---")

# Variation
st.sidebar.subheader("Variation Group")
users_variation = st.sidebar.number_input("Variation Users", min_value=1, value=5000)
conv_variation = st.sidebar.number_input("Variation Conversions", min_value=0, value=600)
rev_variation = st.sidebar.number_input("Variation Revenue ($)", min_value=0.0, value=33000.0)
prod_variation = st.sidebar.number_input("Variation Products Sold", min_value=0, value=1000)

st.sidebar.markdown("---")
days_run = st.sidebar.number_input("Days Test Ran", min_value=1, value=14)

# --- HELPER FUNCTIONS ---

def perform_srm_test(observed_users, expected_split=(0.5, 0.5)):
    total_users = sum(observed_users)
    expected_users = [total_users * p for p in expected_split]
    chi2_stat, p_value = chi2_contingency([observed_users, expected_users])[:2]
    return chi2_stat, p_value

def calculate_uplift(ctrl, var):
    if ctrl == 0: return 0.0
    return ((var - ctrl) / ctrl) * 100

# --- PLOTTING FUNCTIONS ---

def plot_metric_comparison(name, val_c, val_v, unit=""):
    """Generic bar chart for any metric"""
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = ['Control', 'Variation']
    values = [val_c, val_v]
    
    # Dynamic colors
    if val_v >= val_c:
        colors = ['#1f77b4', '#2ca02c'] # Blue, Green
    else:
        colors = ['#1f77b4', '#d62728'] # Blue, Red
        
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
    ax.set_title('Bayesian Probability Density (Conversion Rate)')
    ax.set_xlabel('Conversion Rate')
    ax.legend()
    st.pyplot(fig)

def run_bootstrap_and_plot(users_c, conv_c, users_v, conv_v, n_bootstrap=10000):
    sim_c = np.random.binomial(users_c, conv_c/users_c, n_bootstrap) / users_c
    sim_v = np.random.binomial(users_v, conv_v/users_v, n_bootstrap) / users_v
    diffs = sim_v - sim_c
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(diffs, bins=50, color='orange', edgecolor='black', alpha=0.7)
    lower_ci = np.percentile(diffs, 2.5)
    upper_ci = np.percentile(diffs, 97.5)
    ax.axvline(lower_ci, color='red', linestyle='--', label='95% CI Lower')
    ax.axvline(upper_ci, color='red', linestyle='--', label='95% CI Upper')
    ax.set_title('Bootstrap Distribution of Differences')
    ax.legend()
    st.pyplot(fig)
    return sim_c * 100, sim_v * 100, lower_ci * 100, upper_ci * 100

def plot_box_plot_analysis(sim_samples_c, sim_samples_v):
    data = [sim_samples_c, sim_samples_v]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    medianprops=dict(color="blue", linewidth=1.5),
                    boxprops=dict(facecolor="skyblue", color="blue"),
                    whiskerprops=dict(color="blue"),
                    capprops=dict(color="blue"))
    ax.set_xticklabels(['Control Group', 'Variation Group'])
    ax.set_ylabel('Conversion Rate (%)')
    ax.set_title('Box Plot Analysis')
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

# --- METRIC CALCULATIONS ---

# 1. Conversion Rate
rate_c = conv_control / users_control
rate_v = conv_variation / users_variation
uplift_cr = calculate_uplift(rate_c, rate_v)

# 2. Revenue Metrics (RPV, AOV)
aov_c = rev_control / conv_control if conv_control > 0 else 0
aov_v = rev_variation / conv_variation if conv_variation > 0 else 0
uplift_aov = calculate_uplift(aov_c, aov_v)

rpv_c = rev_control / users_control
rpv_v = rev_variation / users_variation
uplift_rpv = calculate_uplift(rpv_c, rpv_v)

# 3. Product Metrics (Avg Products/Order, Avg Products/User)
apo_c = prod_control / conv_control if conv_control > 0 else 0
apo_v = prod_variation / conv_variation if conv_variation > 0 else 0
uplift_apo = calculate_uplift(apo_c, apo_v)

apu_c = prod_control / users_control
apu_v = prod_variation / users_variation
uplift_apu = calculate_uplift(apu_c, apu_v)

# --- STATISTICAL TESTS ---

# Z-Test for Conversion Rate
z_stat, p_value_z = proportions_ztest([conv_control, conv_variation], [users_control, users_variation])

# SRM Test
chi2_val, p_value_srm = perform_srm_test([users_control, users_variation])

# --- DASHBOARD ---

st.header("Results Summary")

# Metric Group 1: The Main KPIs (CR, RPV, AOV)
st.subheader("1. Primary KPIs")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Conversion Rate", f"{rate_v*100:.2f}%", f"{uplift_cr:+.2f}%")
col2.metric("RPV (Rev/Visitor)", f"${rpv_v:.2f}", f"{uplift_rpv:+.2f}%")
col3.metric("AOV (Avg Order)", f"${aov_v:.2f}", f"{uplift_aov:+.2f}%")
if p_value_z <= 0.05:
    col4.success(f"CR Sig: YES (p={p_value_z:.4f})")
else:
    col4.info(f"CR Sig: NO (p={p_value_z:.4f})")

# Metric Group 2: Product Velocity
st.subheader("2. Product Velocity")
col1, col2, col3 = st.columns(3)
col1.metric("Avg Products / Order", f"{apo_v:.2f}", f"{uplift_apo:+.2f}%")
col2.metric("Avg Products / User", f"{apu_v:.2f}", f"{uplift_apu:+.2f}%")
col3.caption("Higher 'Products / Order' means users are building bigger baskets.")

st.markdown("---")

# --- INTERPRETATION SECTIONS ---

st.subheader("3. Executive Interpretation")
rev_diff = rev_variation - (rev_control * (users_variation/users_control)) # Normalized revenue diff

# Logic for mixed results
if uplift_cr > 0 and uplift_aov < 0:
    st.warning("⚠️ **Trade-off Detected:** Variation drives MORE orders, but SMALLER baskets.")
elif uplift_cr < 0 and uplift_aov > 0:
    st.warning("⚠️ **Trade-off Detected:** Variation drives FEWER orders, but BIGGER baskets.")
else:
    st.success("✅ **Clean Result:** Trends are consistent.")

if uplift_rpv > 0:
    st.write(f"**Financial Impact:** The Variation generates **${rpv_v - rpv_c:.2f} more per visitor**.")
    st.write(f"Projected impact per 100k users: **+${(rpv_v - rpv_c)*100000:,.0f}**")
else:
    st.write(f"**Financial Impact:** The Variation generates **${rpv_c - rpv_v:.2f} less per visitor**.")

# Health Checks
st.subheader("4. Health Checks")
col1, col2 = st.columns(2)
with col1:
    if p_value_srm < 0.01:
        st.error(f"❌ **SRM DETECTED (p={p_value_srm:.4f})**")
    else:
        st.success(f"✅ **SRM PASSED (p={p_value_srm:.4f})**")

with col2:
    if days_run < 14:
        st.warning(f"⚠️ **Too Short ({days_run} Days)**")
    else:
        st.success(f"✅ **Duration OK ({days_run} Days)**")

st.markdown("---")

# --- VISUALIZATIONS ---
st.header("Visualizations")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Product Metrics", "Revenue Charts", "CR Comparison", "Bayesian", "Bootstrap", "Box Plot"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        plot_metric_comparison("Avg Products / Order", apo_c, apo_v, unit="")
    with col2:
        plot_metric_comparison("Avg Products / User", apu_c, apu_v, unit="")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        plot_metric_comparison("Revenue Per Visitor (RPV)", rpv_c, rpv_v, unit="$")
    with col2:
        plot_metric_comparison("Average Order Value (AOV)", aov_c, aov_v, unit="$")

with tab3:
    plot_metric_comparison("Conversion Rate", rate_c*100, rate_v*100, unit="")

with tab4:
    plot_bayesian_pdfs(conv_control+1, users_control-conv_control+1, 
                       conv_variation+1, users_variation-conv_variation+1)

with tab5:
    samples_c, samples_v, ci_low, ci_high = run_bootstrap_and_plot(users_control, conv_control, users_variation, conv_variation)
    st.write(f"**95% Confidence Interval:** {ci_low:.2f}% to {ci_high:.2f}%")

with tab6:
    plot_box_plot_analysis(samples_c, samples_v)
