import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, chi2_contingency
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
import openai

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="A/B Test Analyzer", page_icon="📊", layout="wide")
st.title("A/B Test Analyzer v0.9")

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

def get_ai_analysis(api_key, hypothesis, metrics_dict, provider="OpenAI"):
    if not api_key:
        return "⚠️ Please enter a valid API Key to generate this analysis."
    
    # CONFIGURE PROVIDER
    if provider == "DeepSeek":
        base_url = "https://api.deepseek.com"
        model_name = "deepseek-reasoner"
    else:
        base_url = "https://api.openai.com/v1"
        model_name = "gpt-4o"

    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        
        prompt = f"""
        You are a Senior Data Scientist at a top Conversion Rate Optimization (CRO) agency.
        Analyze the results of this A/B test for a stakeholder presentation.

        **The Hypothesis:** "{hypothesis}"

        **The Data:**
        - Duration: {metrics_dict['days']} days
        - Control Users: {metrics_dict['users_c']}, Variation Users: {metrics_dict['users_v']}
        - SRM P-Value: {metrics_dict['p_srm']:.4f}
        
        **Metrics:**
        - CR: {metrics_dict['cr_c']:.2f}% vs {metrics_dict['cr_v']:.2f}% (Uplift: {metrics_dict['uplift_cr']:.2f}%, p={metrics_dict['p_cr']:.4f})
        - AOV: ${metrics_dict['aov_c']:.2f} vs ${metrics_dict['aov_v']:.2f} (Uplift: {metrics_dict['uplift_aov']:.2f}%)
        - RPV: ${metrics_dict['rpv_c']:.2f} vs ${metrics_dict['rpv_v']:.2f} (Uplift: {metrics_dict['uplift_rpv']:.2f}%)
        - 95% Confidence Interval (Absolute Diff): {metrics_dict['ci_low']:.2f}% to {metrics_dict['ci_high']:.2f}%
        
        **Task:**
        1. Executive Summary (Win/Loss/Inconclusive).
        2. Trade-off Analysis (Volume vs Value).
        3. Final Recommendation (Roll out/Roll back).
        
        **4. Visual Analysis (Graph-by-Graph):**
        Please write 1-2 paragraphs interpreting EACH of the following graphs based on the data above:
        - **Strategic Matrix:** Interpret the position of the dots (CR vs AOV trade-off).
        - **Product Metrics:** Analyze the "Avg Products per Order" and "Avg Products per User".
        - **Revenue Charts:** Compare RPV and AOV bars.
        - **CR Comparison:** Interpret the simple bar chart difference.
        - **Bayesian:** Describe the probability density curves (overlap means uncertainty, separation means confidence).
        - **Bootstrap:** Interpret the histogram of differences using the CI ({metrics_dict['ci_low']:.2f}% to {metrics_dict['ci_high']:.2f}%).
        - **Box Plot:** Interpret the spread/variability of the conversion rates.
        """

        response = client.chat.completions.create(
            model=model_name, 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to AI: {str(e)}"

def generate_smart_analysis(hypothesis, metrics):
    report = []
    
    # 1. Executive Summary
    report.append("### 📄 Executive Summary")
    if metrics['p_srm'] < 0.01:
        report.append(f"❌ **CRITICAL FAILURE:** Sample Ratio Mismatch (SRM) detected (p={metrics['p_srm']:.4f}). Results likely invalid.")
        return "\n\n".join(report) 

    if metrics['p_cr'] <= 0.05:
        if metrics['uplift_cr'] > 0:
            status = "WINNING"
            report.append(f"The Variation is a **STATISTICALLY SIGNIFICANT WINNER**.")
        else:
            status = "LOSING"
            report.append(f"The Variation is a **STATISTICALLY SIGNIFICANT LOSER**.")
    else:
        status = "INCONCLUSIVE"
        report.append(f"The test is **INCONCLUSIVE** (p={metrics['p_cr']:.4f}).")

    # 2. Data Health
    report.append("### 🛡️ Data Health & Validity")
    if metrics['days'] < 7:
        report.append(f"⚠️ **Duration Warning:** Test ran for {metrics['days']} days (too short).")
    elif metrics['days'] < 14:
        report.append(f"⚠️ **Duration Caution:** Test ran for {metrics['days']} days (watch for novelty effects).")
    else:
        report.append(f"✅ **Duration:** Healthy ({metrics['days']} days).")
    report.append(f"✅ **SRM Check:** Passed (p={metrics['p_srm']:.4f}).")

    # 3. Performance Breakdown
    report.append("### 🔍 Performance Breakdown")
    report.append(f"- **Conversion Rate:** {'Improved' if metrics['uplift_cr'] > 0 else 'Decreased'} by **{metrics['uplift_cr']:.2f}%**.")
    
    if abs(metrics['uplift_aov']) < 1.0:
        aov_text = "remained stable"
    else:
        aov_text = f"{'increased' if metrics['uplift_aov'] > 0 else 'decreased'} by {metrics['uplift_aov']:.2f}%"
    report.append(f"- **Average Order Value:** {aov_text} (${metrics['aov_c']:.2f} vs ${metrics['aov_v']:.2f}).")

    # 4. Recommendation
    report.append("### 🚀 Recommendation")
    if metrics['uplift_rpv'] > 0:
        financial_impact = (metrics['rpv_v'] - metrics['rpv_c']) * 100000
        report.append(f"**Financial Outlook: POSITIVE.**")
        report.append(f"Variation generates **${metrics['rpv_v'] - metrics['rpv_c']:.2f} more revenue per visitor**.")
    else:
        report.append(f"**Financial Outlook: NEGATIVE.**")
        report.append(f"Variation generates **${metrics['rpv_c'] - metrics['rpv_v']:.2f} less** per visitor.")

    return "\n\n".join(report)

def calculate_bayesian_risk(alpha_c, beta_c, alpha_v, beta_v, num_samples=50000):
    samples_c = np.random.beta(alpha_c, beta_c, num_samples)
    samples_v = np.random.beta(alpha_v, beta_v, num_samples)
    prob_v_wins = np.mean(samples_v > samples_c)
    loss_v = np.mean(np.maximum(samples_c - samples_v, 0))
    loss_c = np.mean(np.maximum(samples_v - samples_c, 0))
    return prob_v_wins, loss_v, loss_c

# --- PLOTTING FUNCTIONS ---

def plot_strategic_matrix(cr_c, aov_c, cr_v, aov_v):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cr_c, aov_c, color='blue', s=200, label='Control', zorder=5)
    ax.scatter(cr_v, aov_v, color='green', s=200, label='Variation', zorder=5)
    ax.annotate("", xy=(cr_v, aov_v), xytext=(cr_c, aov_c), arrowprops=dict(arrowstyle="->", color='gray', lw=1.5, ls='--'))
    ax.axvline(cr_c, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(aov_c, color='gray', linestyle=':', alpha=0.5)
    ax.annotate("Control", (cr_c, aov_c), xytext=(0, 10), textcoords='offset points', ha='center', fontweight='bold', color='blue')
    ax.annotate("Variation", (cr_v, aov_v), xytext=(0, 10), textcoords='offset points', ha='center', fontweight='bold', color='green')
    ax.set_title("Strategic Matrix: Volume (CR) vs Value (AOV)")
    ax.set_xlabel("Conversion Rate (%)")
    ax.set_ylabel("Average Order Value ($)")
    ax.grid(True, alpha=0.3)
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
    ax.set_title('Bootstrap Distribution')
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

rate_c = conv_control / users_control
rate_v = conv_variation / users_variation
uplift_cr = calculate_uplift(rate_c, rate_v)

aov_c = rev_control / conv_control if conv_control > 0 else 0
aov_v = rev_variation / conv_variation if conv_variation > 0 else 0
uplift_aov = calculate_uplift(aov_c, aov_v)

rpv_c = rev_control / users_control
rpv_v = rev_variation / users_variation
uplift_rpv = calculate_uplift(rpv_c, rpv_v)

apo_c = prod_control / conv_control if conv_control > 0 else 0
apo_v = prod_variation / conv_variation if conv_variation > 0 else 0
uplift_apo = calculate_uplift(apo_c, apo_v)

apu_c = prod_control / users_control
apu_v = prod_variation / users_variation
uplift_apu = calculate_uplift(apu_c, apu_v)

z_stat, p_value_z = proportions_ztest([conv_control, conv_variation], [users_control, users_variation])
chi2_val, p_value_srm = perform_srm_test([users_control, users_variation])

# --- DASHBOARD ---

st.header("Results Summary")
st.subheader("1. Primary KPIs")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Conversion Rate", f"{rate_v*100:.2f}%", f"{uplift_cr:+.2f}%")
col2.metric("RPV (Rev/Visitor)", f"${rpv_v:.2f}", f"{uplift_rpv:+.2f}%")
col3.metric("AOV (Avg Order)", f"${aov_v:.2f}", f"{uplift_aov:+.2f}%")
if p_value_z <= 0.05:
    col4.success(f"CR Sig: YES (p={p_value_z:.4f})")
else:
    col4.info(f"CR Sig: NO (p={p_value_z:.4f})")

st.subheader("2. Product Velocity")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Avg Products / Order", f"{apo_v:.2f}", f"{uplift_apo:+.2f}%")
col2.metric("Avg Products / User", f"{apu_v:.2f}", f"{uplift_apu:+.2f}%")
col4.caption("Higher 'Products / Order' means users are building bigger baskets.")

st.markdown("---")

st.subheader("3. Executive Interpretation")
if uplift_cr > 0 and uplift_aov < 0:
    st.warning("⚠️ **Trade-off Detected:** Variation drives MORE orders, but SMALLER baskets.")
elif uplift_cr < 0 and uplift_aov > 0:
    st.warning("⚠️ **Trade-off Detected:** Variation drives FEWER orders, but BIGGER baskets.")
else:
    st.success("✅ **Clean Result:** Trends are consistent.")

if uplift_rpv > 0:
    st.write(f"**Financial Impact:** Variation generates **${rpv_v - rpv_c:.2f} more** per visitor.")
else:
    st.write(f"**Financial Impact:** Variation generates **${rpv_c - rpv_v:.2f} less** per visitor.")

st.subheader("4. Health Checks")
col1, col2 = st.columns(2)
with col1:
    if p_value_srm < 0.01:
        st.error(f"❌ **SRM DETECTED (p={p_value_srm:.4f})**")
    else:
        st.success(f"✅ **SRM PASSED (p={p_value_srm:.4f})**")

with col2:
    if days_run < 7:
        st.error(f"❌ **Too Short ({days_run} Days)**")
    elif days_run < 14:
        st.warning(f"⚠️ **Short Duration ({days_run} Days)**")
    else:
        st.success(f"✅ **Duration OK ({days_run} Days)**")

st.markdown("---")

# --- TABS ---
st.header("Deep Dive Analysis")
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🧠 Smart Analysis", 
    "🤖 AI Analysis", 
    "🛑 Stopping & Sequential", 
    "Strategic Matrix", 
    "Product Metrics", 
    "Revenue Charts", 
    "CR Comparison", 
    "Bayesian", 
    "Bootstrap", 
    "Box Plot"
])

# --- TAB 1: SMART ANALYSIS ---
with tab1:
    st.markdown("### 🧠 Smart Executive Summary (Free)")
    st.info("Generated instantly using statistical rules (No API Key required).")
    user_hypothesis = st.text_area("Hypothesis:", placeholder="We believed that...", height=70, key="hyp_smart")
    if st.button("Generate Smart Analysis"):
        metrics_payload = {
            "days": days_run, "users_c": users_control, "users_v": users_variation, "p_srm": p_value_srm,
            "cr_c": rate_c*100, "cr_v": rate_v*100, "uplift_cr": uplift_cr, "p_cr": p_value_z,
            "aov_c": aov_c, "aov_v": aov_v, "uplift_aov": uplift_aov, "rpv_c": rpv_c, "rpv_v": rpv_v, "uplift_rpv": uplift_rpv
        }
        smart_result = generate_smart_analysis(user_hypothesis, metrics_payload)
        st.markdown("---")
        st.markdown(smart_result)
        
        st.markdown("### 📋 Copy Report")
        st.code(smart_result, language='markdown')

# --- TAB 2: AI ANALYSIS ---
with tab2:
    st.markdown("### 🤖 AI-Powered Analysis")
    col_prov, col_key = st.columns(2)
    with col_prov:
        ai_provider = st.selectbox("Select Provider", ["OpenAI (GPT-4o)", "DeepSeek (R1)"])
    with col_key:
        api_key_input = st.text_input("Enter API Key", type="password")
    
    user_hypothesis_ai = st.text_area("Hypothesis (AI):", placeholder="We believed that...", height=100, key="hyp_ai")
    
    if st.button("Generate AI Analysis"):
        ci_low, ci_high = proportion_confint(conv_variation, users_variation, alpha=0.05, method='normal')
        diff_ci_low = (ci_low - rate_c) * 100
        diff_ci_high = (ci_high - rate_c) * 100

        metrics_payload = {
            "days": days_run, "users_c": users_control, "users_v": users_variation, "p_srm": p_value_srm,
            "cr_c": rate_c*100, "cr_v": rate_v*100, "uplift_cr": uplift_cr, "p_cr": p_value_z,
            "aov_c": aov_c, "aov_v": aov_v, "uplift_aov": uplift_aov, "rpv_c": rpv_c, "rpv_v": rpv_v, "uplift_rpv": uplift_rpv,
            "ci_low": diff_ci_low, "ci_high": diff_ci_high
        }
        
        provider_name = "DeepSeek" if "DeepSeek" in ai_provider else "OpenAI"
        with st.spinner(f"Connecting to {provider_name}..."):
            ai_result = get_ai_analysis(api_key_input, user_hypothesis_ai, metrics_payload, provider=provider_name)
            st.markdown("---")
            st.markdown(ai_result)
            
            st.markdown("### 📋 Copy Report")
            st.code(ai_result, language='markdown')

# --- TAB 3: STOPPING & SEQUENTIAL ---
with tab3:
    st.markdown("### 🛑 Stopping Rules & Sequential Testing")
    prob_v_wins, loss_v, loss_c = calculate_bayesian_risk(
        conv_control+1, users_control-conv_control+1, 
        conv_variation+1, users_variation-conv_variation+1
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Probability Best", f"{prob_v_wins*100:.1f}%")
    c2.metric("Risk (Switch)", f"{loss_v*100:.5f}%")
    c3.metric("Risk (Stay)", f"{loss_c*100:.5f}%")
    
    st.markdown("---")
    st.markdown("#### 2. Sequential Testing (Peeking Penalty)")
    peeks = st.number_input("How many times have you checked the results?", min_value=1, value=1)
    adjusted_alpha = 0.05 * np.log(1 + (np.e - 1) / peeks)
    st.write(f"Adjusted significance threshold: **{adjusted_alpha:.4f}**")
    if p_value_z < adjusted_alpha:
        st.success(f"✅ **SIGNIFICANT** (p={p_value_z:.4f})")
    else:
        st.error(f"❌ **NOT SIGNIFICANT** (p={p_value_z:.4f})")

# --- OTHER TABS ---
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
    samples_c, samples_v, ci_low, ci_high = run_bootstrap_and_plot(users_control, conv_control, users_variation, conv_variation)
    st.write(f"**95% CI:** {ci_low:.2f}% to {ci_high:.2f}%")
with tab10: plot_box_plot_analysis(samples_c, samples_v)
