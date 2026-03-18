# Enterprise A/B Test Analyzer

A single-file Streamlit application for analysing A/B and A/B/n experiments. Built for CRO specialists, product managers, and data analysts who work with aggregate experiment data — no raw event logs required.

---

## Features

### Statistical Engine
- **Frequentist analysis** — two-proportion z-test, chi-square omnibus test, confidence intervals
- **Multiple comparison correction** — Holm, Bonferroni, or Benjamini-Hochberg (FDR) for A/B/n tests
- **Bayesian analysis** — Monte Carlo Beta posteriors, P(best), and Expected Loss across all groups
- **Revenue significance** — Mann-Whitney U + bootstrapped log-normal reconstruction for AOV and RPV
- **SRM detection** — Chi-square goodness-of-fit test flags Sample Ratio Mismatch automatically

### Goal-Based Winner Selection
Three optimisation modes drive winner logic with a 4-level tiebreaker chain:

| Mode | Primary criterion |
|---|---|
| Maximize CR | Highest conversion rate uplift |
| Maximize Revenue | Highest RPV uplift (a lower-CR variant can win if revenue is higher) |
| Balanced | Composite score: 40% CR + 60% RPV |

### Guardrail Metrics
Up to 3 secondary metrics (e.g. bounce rate, session duration, add-to-cart rate) with user-defined direction and max allowed change %. Shows pass/fail status alongside primary results so "what you broke" is as visible as "what you won."

### Multi-Variant Support (A/B/n)
Up to 4 groups (Control + 3 variations). Omnibus chi-square test plus pairwise comparisons with multiple comparison correction. All charts and tables scale automatically.

### Reporting
- **Smart Analysis** — instant rule-based report covering significance, revenue, product metrics, guardrails, and recommendation. No API key needed.
- **AI Analysis** — structured LLM report via OpenAI (GPT-4o) or DeepSeek (R1). API key entered at runtime, never saved.

### Sample Size Planning
Built-in calculator for required sample size, estimated test duration, and MDE power curves.

### Health Checks
- SRM badge with chi-square result
- Duration warnings (too short, novelty effect risk)
- Conversion rate sanity check (warns if conv > users)

### Data Visualisations (10 deep-dive tabs)
Smart Analysis, AI Analysis, Stopping & Sequential, Strategic Matrix (CR vs AOV), Product Metrics (APO/APU), Revenue Charts, CR Comparison, Bayesian PDFs, Bootstrap CI histogram, Box plots. All charts use a dark theme matching Streamlit's UI.

### PDF Export
Click **Prepare PDF Report** in the sidebar to generate a multi-page PDF containing: cover page, key metrics table, statistical results, Bayesian results, revenue significance, health checks, 4 embedded charts (CR, Strategic Matrix, Bayesian PDFs, Bootstrap CI), and the Smart Analysis report if previously generated. Charts use a light theme optimised for print.

### Save & Load
Export current experiment data as a JSON snapshot and reload it later. All inputs — including guardrail metrics and goal settings — are preserved.

---

## Installation

Requires Python 3.9+.

```bash
pip install streamlit numpy pandas scipy statsmodels matplotlib openai reportlab
streamlit run ab_test_analyzer.py
```

The app opens at `http://localhost:8501`. No `.env` file or pre-configured API keys needed.

---

## How to Use

### 1. Settings (sidebar)
- Set **days the test ran** and **confidence level** (90 / 95 / 99%)
- Choose your **primary goal** (Maximize CR / Maximize Revenue / Balanced)
- Optionally configure **guardrail metrics** (up to 3)
- For A/B/n tests, choose a **multiple comparison correction** method

### 2. Enter Results (sidebar)
- Select number of variations (1–3)
- Enter users, conversions, revenue, and products sold for each group

### 3. Read the Dashboard
- **Top KPIs** — uplift, significance, winner declaration
- **Health Checks** — SRM, duration, guardrail pass/fail
- **Deep Dive Tabs** — explore statistical details, charts, and reports

### 4. Generate a Report
- **Smart Analysis tab** — enter your hypothesis and click Generate (free, no key)
- **AI Analysis tab** — select provider, enter API key, click Generate

---

## Statistical Methodology

| Method | Implementation |
|---|---|
| Significance test | Two-proportion z-test (`proportions_ztest`) |
| Omnibus test | Pearson chi-square (`chi2_contingency`) |
| MCC | `multipletests` from statsmodels (Holm / Bonferroni / FDR-BH) |
| SRM check | Chi-square goodness-of-fit (`chisquare`) — not independence test |
| Bayesian | Monte Carlo, 50,000 draws from Beta(conv+1, non-conv+1) |
| Revenue sig | Mann-Whitney U + bootstrapped log-normal (2,000 resamples, seed 42) |
| Bootstrap CI | 10,000 binomial resamples |
| Guardrails | Threshold-based % change — no p-values (standard CRO practice) |

---

## Disclaimer

This tool provides statistical guidance and should not be the sole basis for critical business decisions. Always validate data integrity before analysis. Revenue significance uses reconstructed distributions from aggregates — treat results as directional signals, not precise p-values.
