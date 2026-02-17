import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, chi2_contingency
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.power import NormalIndPower
import openai

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="A/B Test Analyzer", page_icon="📊", layout="wide")
st.title("A/B Test Analyzer v1.5")

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

# --- SIDEBAR: PLANNING TOOL ---
st.sidebar.markdown("---")
with st.sidebar.expander("📅 Test Planning (Sample Size)"):
    st.caption("Estimate required sample size.")
    baseline_cr = st.number_input("Baseline CR (%)", value=10.0, step=0.1)
    mde = st.number_input("Min Detectable Effect (%)", value=5.0, step=0.1)
    power = 0.8
    alpha = 0.05
    
    if st.button("Calculate Needed Users"):
        effect_size = proportion_effectsize(baseline_cr/100, (baseline_cr * (1 + mde/100))/100)
        sample_size = NormalIndPower().solve_power(effect_size, power=power, alpha=alpha, ratio=1)
        st.write(f"**{int(sample_size):,} users** per group.")

# --- HELPER FUNCTIONS ---

def perform_srm_test(observed_users, expected_split=(0.5, 0.5)):
    total_users = sum(observed_users)
    expected_users = [total_users * p for p in expected_split]
    chi2_stat, p_value = chi2_contingency([observed_users, expected_users])[:2]
    return chi2_stat, p_value

def calculate_uplift(ctrl, var):
    if ctrl == 0: return 0.0
    return ((var - ctrl) / ctrl) * 100

def calculate_bayesian_risk(alpha_c, beta_c, alpha_v, beta_v, num_samples=50000):
    samples_c = np.random.beta(alpha_c, beta_c, num_samples)
    samples_v = np.random.beta(alpha_v, beta_v, num_samples)
    prob_v_wins = np.mean(samples_v > samples_c)
    loss_v = np.mean(np.maximum(samples_c - samples_v, 0))
    loss_c = np.mean(np.maximum(samples_v - samples_c, 0))
    return prob_v_wins, loss_v, loss_c

def check_simpsons_paradox(seg1_data, seg2_data):
    """
    Checks if the direction of the segments contradicts the direction of the aggregate.
    """
    # Calculate Segment 1 Uplift
    cr_c1 = seg1_data['conv_c'] / seg1_data['users_c']
    cr_v1 = seg1_data['conv_v'] / seg1_data['users_v']
    uplift_1 = calculate_uplift(cr_c1, cr_v1)
    
    # Calculate Segment 2 Uplift
    cr_c2 = seg2_data['conv_c'] / seg2_data['users_c']
    cr_v2 = seg2_data['conv_v'] / seg2_data['users_v']
    uplift_2 = calculate_uplift(cr_c2, cr_v2)
    
    # Calculate Aggregate Uplift (Sum of inputs)
    agg_users_c = seg1_data['users_c'] + seg2_data['users_c']
    agg_conv_c = seg1_data['conv_c'] + seg2_data['conv_c']
    agg_users_v = seg1_data['users_v'] + seg2_data['users_v']
    agg_conv_v = seg1_data['conv_v'] + seg2_data['conv_v']
    
    cr_c_agg = agg_conv_c / agg_users_c
    cr_v_agg = agg_conv_v / agg_users_v
    uplift_agg = calculate_uplift(cr_c_agg, cr_v_agg)
    
    # Check Logic: If both segments are Positive but Agg is Negative (or vice versa)
    paradox = False
    if (uplift_1 > 0 and uplift_2 > 0 and uplift_agg < 0):
        paradox = True
    elif (uplift_1 < 0 and uplift_2 < 0 and uplift_agg > 0):
        paradox = True
        
    return paradox, uplift_1, uplift_2, uplift_agg

def get_ai_analysis(api_key, hypothesis, metrics_dict, provider="OpenAI"):
    if not api_key:
        return "⚠️ Please enter a valid API Key."
    if provider == "DeepSeek":
        base_url = "https://api.deepseek.com"
        model_name = "deepseek-reasoner"
    else:
        base_url = "https://api.openai.com/v1"
        model_name = "gpt-4o"
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        prompt = f"""
        Analyze this A/B test. No memo header.
        Hypothesis: "{hypothesis}"
        Metrics: CR Uplift: {metrics_dict['uplift_cr']:.2f}% (p={metrics_dict['p_cr']:.4f}), RPV Uplift: {metrics_dict['uplift_rpv']:.2f}%.
        Bayesian: Prob Best: {metrics_dict['prob_v_wins']:.1f}%, Risk Switch: {metrics_dict['loss_v']:.5f}%.
        Task: Executive Summary, Trade-offs, Bayesian Risk, Recommendation.
        Visuals: Interpret Strategic Matrix, Bootstrap, Box Plot.
        """
        response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def generate_smart_analysis(hypothesis, metrics):
    report = ["### 📄 Executive Summary"]
    if metrics['p_srm'] < 0.01:
        return "❌ **CRITICAL FAILURE:** SRM detected."
    if metrics['p_cr'] <= 0.05:
        report.append(f"The Variation is a **{'WINNER' if metrics['uplift_cr'] > 0 else 'LOSER'}**.")
    else:
        report.append("The test is **INCONCLUSIVE**.")
    report.append(f"**CR Uplift:** {metrics['uplift_cr']:.2f}% | **RPV Uplift:** {metrics['uplift_rpv']:.2f}%")
    return "\n\n".join(report)

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
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Conversion Rate", f"{rate_v*100:.2f}%", f"{uplift_cr:+.2f}%")
col2.metric("RPV (Rev/Visitor)", f"${rpv_v:.2f}", f"{uplift_rpv:+.2f}%")
col3.metric("AOV (Avg Order)", f"${aov_v:.2f}", f"{uplift_aov:+.2f}%")
col4.success(f"Sig: YES (p={p_value_z:.4f})") if p_value_z <= 0.05 else col4.info(f"Sig: NO (p={p_value_z:.4f})")

st.markdown("---")

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🧠 Smart Analysis", "🤖 AI Analysis", "🕵️ Segments & Simpson's", "🛑 Stopping & Sequential", 
    "Strategic Matrix", "Product Metrics", "Revenue Charts", "CR Comparison", "Bayesian", "Bootstrap"
])

# --- TAB 1 & 2: ANALYSIS (Simplified for brevity in this view) ---
with tab1:
    st.markdown("### 🧠 Smart Analysis")
    if st.button("Generate Smart Report"):
        st.write(generate_smart_analysis("Hypothesis", {'p_srm': p_value_srm, 'p_cr': p_value_z, 'uplift_cr': uplift_cr, 'uplift_rpv': uplift_rpv}))
with tab2:
    st.markdown("### 🤖 AI Analysis")
    api_key = st.text_input("API Key", type="password")
    if st.button("Generate AI Report") and api_key:
        st.write("Generating...")

# --- TAB 3: SEGMENTS & SIMPSON'S (NEW!) ---
with tab3:
    st.markdown("### 🕵️ Simpson's Paradox Detector")
    st.info("Input data for two segments (e.g., Mobile vs Desktop) to check if the aggregate result is misleading.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Segment 1 (e.g. Mobile)")
        s1_name = st.text_input("Name", "Mobile")
        s1_users_c = st.number_input("Control Users", 2000, key="s1uc")
        s1_conv_c = st.number_input("Control Conv", 100, key="s1cc")
        s1_users_v = st.number_input("Var Users", 2000, key="s1uv")
        s1_conv_v = st.number_input("Var Conv", 120, key="s1cv")
        
    with c2:
        st.subheader("Segment 2 (e.g. Desktop)")
        s2_name = st.text_input("Name", "Desktop")
        s2_users_c = st.number_input("Control Users", 3000, key="s2uc")
        s2_conv_c = st.number_input("Control Conv", 400, key="s2cc")
        s2_users_v = st.number_input("Var Users", 3000, key="s2uv")
        s2_conv_v = st.number_input("Var Conv", 380, key="s2cv")

    if st.button("Check for Paradox"):
        seg1 = {'users_c': s1_users_c, 'conv_c': s1_conv_c, 'users_v': s1_users_v, 'conv_v': s1_conv_v}
        seg2 = {'users_c': s2_users_c, 'conv_c': s2_conv_c, 'users_v': s2_users_v, 'conv_v': s2_conv_v}
        
        is_paradox, up1, up2, up_agg = check_simpsons_paradox(seg1, seg2)
        
        st.markdown("---")
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric(f"{s1_name} Uplift", f"{up1:.2f}%")
        col_res2.metric(f"{s2_name} Uplift", f"{up2:.2f}%")
        col_res3.metric("Aggregate Uplift", f"{up_agg:.2f}%")
        
        if is_paradox:
            st.error(f"⚠️ **SIMPSON'S PARADOX DETECTED!**")
            st.write(f"The Variation is winning in both **{s1_name}** and **{s2_name}**, but it appears to be LOSING overall.")
            st.write("This happens because of a traffic mismatch (SRM) or uneven distribution between high/low performing segments.")
            st.warning("👉 **Trust the Segment data, not the Aggregate data.**")
        else:
            st.success("✅ **No Paradox Detected.** The trends are consistent.")

# --- OTHER TABS (Standard Plots) ---
with tab4: # Stopping
    prob_v, loss_v, loss_c = calculate_bayesian_risk(conv_control+1, users_control-conv_control+1, conv_variation+1, users_variation-conv_variation+1)
    st.metric("Probability Best", f"{prob_v*100:.1f}%")
    
# (Visualization plots omitted for brevity, they remain same as previous version)
