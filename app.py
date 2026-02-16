import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, chi2_contingency
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.power import NormalIndPower

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="A/B Test Analyzer", page_icon="📊", layout="wide")
st.title("📊 Professional A/B Test Analyzer")

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("Experiment Data")
users_control = st.sidebar.number_input("Control Users", min_value=1, value=5000)
conv_control = st.sidebar.number_input("Control Conversions", min_value=0, value=500)
users_variation = st.sidebar.number_input("Variation Users", min_value=1, value=5000)
conv_variation = st.sidebar.number_input("Variation Conversions", min_value=0, value=600)
days_run = st.sidebar.number_input("Days Test Ran", min_value=1, value=14)

# --- HELPER FUNCTIONS ---

def perform_srm_test(observed_users, expected_split=(0.5, 0.5)):
    total_users = sum(observed_users)
    expected_users = [total_users * p for p in expected_split]
    chi2_stat, p_value = chi2_contingency([observed_users, expected_users])[:2]
    return chi2_stat, p_value

# --- PLOTTING FUNCTIONS (Adapted for Streamlit) ---

def plot_cumulative_conversion_comparison(control_rate, variation_rate):
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = ['Control', 'Variation']
    rates = [control_rate, variation_rate]
    colors = ['#1f77b4', '#2ca02c']
    ax.bar(groups, rates, color=colors, alpha=0.8)
    ax.set_title('Conversion Rate Comparison')
    ax.set_ylabel('Conversion Rate (%)')
    ax.set_ylim(0, max(rates) * 1.15)
    for i, rate in enumerate(rates):
        ax.text(i, rate + (max(rates)*0.01), f"{rate:.2f}%", ha='center', va='bottom', fontweight='bold')
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
    ax.set_xlabel('Conversion Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def plot_cumulative_diff_distribution(alpha_c, beta_c, alpha_v, beta_v):
    samples_c = beta(alpha_c, beta_c).rvs(10000)
    samples_v = beta(alpha_v, beta_v).rvs(10000)
    diff = samples_v - samples_c

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(diff, bins=50, density=True, cumulative=True, color='purple', alpha=0.6, edgecolor='purple')
    ax.set_title('Cumulative Probability of Uplift')
    ax.set_xlabel('Difference in Conversion Rate')
    ax.set_ylabel('Cumulative Probability')
    ax.axvline(x=0, color='red', linestyle='--', label='No Effect (0)')
    ax.legend()
    ax.grid(True, alpha=0.5)
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

# --- MAIN LOGIC ---

# Calculations
rate_c = conv_control / users_control
rate_v = conv_variation / users_variation
uplift = ((rate_v - rate_c) / rate_c) * 100

# Stats
z_stat, p_value_z = proportions_ztest([conv_control, conv_variation], [users_control, users_variation])
chi2_val, p_value_srm = perform_srm_test([users_control, users_variation])

# --- RESULTS DASHBOARD ---

st.header("Results Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Control CR", f"{rate_c*100:.2f}%")
col2.metric("Variation CR", f"{rate_v*100:.2f}%")
col3.metric("Relative Uplift", f"{uplift:+.2f}%", delta_color="normal")

st.markdown("---")

# 1. P-Value
st.subheader("1. Statistical Significance")
if p_value_z <= 0.05:
    st.success(f"✅ **SIGNIFICANT (p={p_value_z:.4f})**")
    st.write(f"There is only a {p_value_z*100:.2f}% chance this happened by luck.")
else:
    st.warning(f"⚠️ **NOT SIGNIFICANT (p={p_value_z:.4f})**")
    st.write(f"There is a {p_value_z*100:.2f}% chance this difference is just random noise.")

# 2. SRM Check
st.subheader("2. Sample Ratio Mismatch (SRM) Check")
if p_value_srm < 0.01:
    st.error(f"❌ **CRITICAL WARNING: SRM DETECTED (p={p_value_srm:.4f})**")
    st.write("The traffic split is NOT balanced. Do not trust these results.")
else:
    st.success(f"✅ **PASS (p={p_value_srm:.4f})**")
    st.write("Traffic split looks healthy.")

# 3. Duration Check
st.subheader("3. Duration Check")
if days_run < 14:
    if p_value_z <= 0.05:
        st.warning(f"⚠️ **Test ran for only {days_run} days.** Even if significant, this might be a 'False Positive'. Recommendation: 14+ days.")
    else:
        st.warning(f"⚠️ **Test ran for only {days_run} days.** Data is likely insufficient.")
else:
    st.success(f"✅ **{days_run} days is a healthy duration.**")

st.markdown("---")

# --- VISUALIZATIONS ---
st.header("Visualizations")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Comparison", "Bayesian", "Cumulative Uplift", "Bootstrap", "Box Plot"])

with tab1:
    plot_cumulative_conversion_comparison(rate_c*100, rate_v*100)

with tab2:
    plot_bayesian_pdfs(conv_control+1, users_control-conv_control+1, 
                       conv_variation+1, users_variation-conv_variation+1)

with tab3:
    plot_cumulative_diff_distribution(conv_control+1, users_control-conv_control+1, 
                                      conv_variation+1, users_variation-conv_variation+1)

with tab4:
    samples_c, samples_v, ci_low, ci_high = run_bootstrap_and_plot(users_control, conv_control, users_variation, conv_variation)
    st.write(f"**95% Confidence Interval:** {ci_low:.2f}% to {ci_high:.2f}%")
    if ci_low > 0:
        st.success("✅ **Positive Outcome:** The entire interval is above 0. Variation wins.")
    elif ci_high < 0:
        st.error("❌ **Negative Outcome:** The entire interval is below 0. Variation loses.")
    else:
        st.warning("⚠️ **Inconclusive:** The interval crosses 0.")

with tab5:
    plot_box_plot_analysis(samples_c, samples_v)
