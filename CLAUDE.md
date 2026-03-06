# CLAUDE.md — Enterprise A/B Test Analyzer

This file documents the codebase for Claude (or any AI assistant) working on this project. Read it fully before making any changes.

---

## Project Overview

A single-file Streamlit application (`ab_test_analyzer.py`) for analysing A/B and A/B/n experiments. It accepts aggregate experiment data (no raw event logs), computes frequentist and Bayesian statistics, detects data quality issues, and generates both rule-based and AI-powered written reports.

**Key constraint:** The entire app is one Python file (~1,040 lines). There is no build step, no database, no separate backend. Everything runs top-to-bottom on each Streamlit rerender.

---

## How to Run

```bash
pip install streamlit numpy pandas scipy statsmodels matplotlib openai
streamlit run ab_test_analyzer.py
```

Requires Python 3.9+. No `.env` file needed — API keys are entered by the user inside the UI at runtime.

---

## File Structure

The file has a strict top-to-bottom layout. Sections must stay in this order because Streamlit executes sequentially and sidebar widgets must be rendered before dashboard code reads their values.

```
1.   Imports
2.   Page config + matplotlib backend
3.   SVG icon constants
4.   Numeric constants (BAYESIAN_SAMPLES, BOOTSTRAP_SAMPLES, etc.)
5.   State initialization  ← initialize_state() called immediately
6.   SAVE_KEYS list
7.   Helper functions (render_header, safe_divide, calculate_uplift)
8.   SRM test function
9.   Multi-variant statistical engine  ← run_multivariate_analysis()
10.  Bayesian functions
11.  Revenue significance functions
12.  Simpson's Paradox function
13.  AI analysis function  ← get_ai_analysis()
14.  Smart analysis function  ← generate_smart_analysis()
15.  Plotting functions
16.  SIDEBAR rendering  ← all st.sidebar.* calls
17.  BUILD GROUPS & RUN ANALYSIS  ← core computation block
18.  MAIN DASHBOARD  ← st.title(), metrics, health checks
19.  Simpson's Paradox expander
20.  DEEP DIVE TABS (tab1–tab10)
```

**Never move sidebar rendering below the computation block.** Streamlit widget values are only available after their `st.*` call executes.

---

## Constants

| Constant | Value | Purpose |
|---|---|---|
| `BAYESIAN_SAMPLES` | 50,000 | Monte Carlo draws per Beta distribution |
| `BOOTSTRAP_SAMPLES` | 10,000 | Binomial resamples per bootstrap run |
| `MAX_VARIATIONS` | 3 | Maximum variation groups (so max 4 total: control + 3) |
| `GROUP_COLORS` | `["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]` | One colour per group: Control, Var A, Var B, Var C |

---

## Session State Keys

All state is managed via `st.session_state`. The `initialize_state()` function sets defaults on first load without overwriting existing values. The `SAVE_KEYS` list controls what gets serialised to the JSON snapshot.

### Input keys

| Key | Type | Description |
|---|---|---|
| `num_variations` | int | 1–3; drives dynamic sidebar input rendering |
| `users_c` / `conv_c` / `rev_c` / `prod_c` | int/float | Control group inputs |
| `users_v0` / `conv_v0` / `rev_v0` / `prod_v0` | int/float | Variation A inputs |
| `users_v1` / `conv_v1` / `rev_v1` / `prod_v1` | int/float | Variation B inputs |
| `users_v2` / `conv_v2` / `rev_v2` / `prod_v2` | int/float | Variation C inputs |
| `days` | int | Test duration in days |
| `conf_level` | str | `"90%"` / `"95%"` / `"99%"` |
| `mc_method` | str | `"holm"` / `"bonferroni"` / `"fdr_bh"` |

### Planning calculator keys
`p_traffic`, `p_base_cr`, `p_base_aov`, `p_mde`, `p_vol`

### Simpson's Paradox keys
`s1_uc`, `s1_cc`, `s1_uv`, `s1_cv`, `s2_uc`, `s2_cc`, `s2_uv`, `s2_cv`

### Internal (computed, not saved)
`_boot_samples_c`, `_boot_samples_v`, `_boot_label_v` — bootstrap results shared between tab9 and tab10

---

## Core Data Structures

### `groups` list
Built just before the main computation block. Every statistical function receives this list.

```python
groups = [
    {"name": "Control",     "users": int, "conv": int, "rev": float, "prod": int},
    {"name": "Variation A", "users": int, "conv": int, "rev": float, "prod": int},
    # ...up to Variation C
]
```

`groups[0]` is always Control. Variations follow in order A, B, C.

### `mv` dict — return value of `run_multivariate_analysis()`

```python
mv = {
    "chi2_global":   float,   # omnibus chi-square statistic
    "p_global":      float,   # omnibus p-value (chi2_contingency across all groups)
    "pairwise":      list,    # one dict per variation (see below)
    "metrics":       list,    # one dict per group including control (see below)
    "winner":        str|None,# name of winning variation, or None
    "correction":    str,     # mc_method used ("holm", "bonferroni", "fdr_bh")
    "n_comparisons": int,     # number of variations tested
}
```

**`mv["pairwise"]` — one entry per variation:**
```python
{
    "name":        str,
    "z_stat":      float,
    "p_raw":       float,   # uncorrected two-sided z-test p-value
    "p_adjusted":  float,   # after MCC via statsmodels.stats.multitest.multipletests
    "significant": bool,    # True if reject[i] from multipletests
    "uplift_cr":   float,   # % uplift in conversion rate vs control
    "uplift_aov":  float,   # % uplift in AOV vs control
    "uplift_rpv":  float,   # % uplift in RPV vs control
}
```

**`mv["metrics"]` — one entry per group (including control):**
```python
{
    "name":       str,
    "users":      int,
    "conv":       int,
    "cr":         float,   # conversion rate (0–1)
    "cr_pct":     float,   # conversion rate × 100
    "aov":        float,   # avg order value (rev / conv)
    "rpv":        float,   # revenue per visitor (rev / users)
    "apo":        float,   # avg products per order (prod / conv)
    "apu":        float,   # avg products per user (prod / users)
    "uplift_cr":  float,   # 0.0 for control, % vs control for variations
    "uplift_aov": float,
    "uplift_rpv": float,
}
```

### `bayes` dict — return value of `calculate_bayesian_multivariate()`
```python
bayes = {
    "prob_best":    {"Control": float, "Variation A": float, ...},  # sum to 1.0
    "expected_loss": {"Control": float, "Variation A": float, ...},  # in rate units
}
```

### `rev_sig` dict — return value of `test_revenue_significance()`
```python
rev_sig = {
    "aov": {
        "mw_p": float, "mw_sig": bool,
        "boot_p": float, "boot_sig": bool,
        "boot_ci_low": float, "boot_ci_high": float,
        "sig": bool,   # True if EITHER test is significant
    },
    "rpv": { ...same structure... }
}
```

---

## Key Functions

### `run_multivariate_analysis(groups, alpha, mc_method)` — lines ~111–198
The central statistical engine. Called once per rerender with the full groups list.

- Computes per-group metrics (CR, AOV, RPV, APO, APU, uplifts)
- Omnibus chi-square via `chi2_contingency` on a (n_groups × 2) contingency table
- Pairwise z-tests via `proportions_ztest` for each variation vs control
- Multiple comparison correction via `multipletests(raw_p_values, method=mc_method)`
- Winner selection: significant variation with highest CR uplift

**Important:** In classic A/B mode (1 variation), `multipletests` with a single p-value is equivalent to no correction. The adjusted p-value equals the raw p-value.

### `calculate_bayesian_multivariate(groups)` — lines ~204–226
Monte Carlo across all groups simultaneously.
- Draws `BAYESIAN_SAMPLES` from `Beta(conv+1, non_conv+1)` for each group
- Stacks into a `(BAYESIAN_SAMPLES, n_groups)` array
- `prob_best[name]` = fraction of draws where that group had the highest CR
- `expected_loss[name]` = expected value of `max(all others) - self` when self is not best

### `test_revenue_significance(uc, cc, rc, uv, cv, rv, ...)` — lines ~255–283
Always compares Control vs the **best variation by CR** only (not all pairs). Revenue significance for multi-variant pairwise comparisons is not implemented — only the top performer is checked.

Reconstructs per-order distributions from aggregates using log-normal (σ=0.8). Uses fixed seed 42 for reproducibility. Results are directional signals, not precise p-values.

### `generate_smart_analysis(hypothesis, mv_results, bayes_mv, metrics_payload, alpha_val)` — lines ~320–470
Rule-based report generator. Takes the pre-computed `mv` and `bayes` dicts directly. Does NOT re-run statistics. Returns a Markdown string rendered directly into tab1.

### `get_ai_analysis(api_key, hypothesis, metrics, provider, conf_level)` — lines ~295–320
Sends a structured prompt to OpenAI GPT-4o or DeepSeek R1 via the OpenAI-compatible client. The prompt explicitly requires 9 Markdown sections. Temperature is 0.7. The `metrics` argument is a dict serialised with `json.dumps(indent=2)` in the caller.

### `perform_srm_test(observed, expected_split=None)` — lines ~98–105
Uses `scipy.stats.chisquare` (goodness-of-fit), **not** `chi2_contingency`. Default assumes equal traffic allocation. Supports N groups.

---

## The Dashboard Computation Block (lines ~706–725)

This is the only place where `groups`, `mv`, `bayes`, `p_srm`, `ctrl_m`, `best_m`, and `rev_sig` are created. Everything below this block reads from these variables.

```python
groups = [ctrl_dict] + var_inputs         # built from sidebar widget values
mv     = run_multivariate_analysis(...)   # all stats
bayes  = calculate_bayesian_multivariate(...)
srm_stat, p_srm = perform_srm_test(...)
ctrl_m = mv["metrics"][0]                 # always Control
best_m = max(mv["metrics"][1:], key=lambda m: m["cr"])  # best variation by CR
best_g = next(g for g in groups if g["name"] == best_m["name"])
rev_sig = test_revenue_significance(ctrl vs best_g only)
```

`best_m` is used as the "focal" variation for bootstrap, box plot, and revenue significance. It is not necessarily the statistically significant winner — it is simply the variation with the highest observed CR.

---

## The 10 Deep Dive Tabs

| Tab | Variable | Content |
|---|---|---|
| tab1 | Smart Analysis | Rule-based report; button-triggered; calls `generate_smart_analysis()` |
| tab2 | AI Analysis | LLM report; button-triggered; calls `get_ai_analysis()` |
| tab3 | Stopping & Sequential | Bayesian P(best) for all groups + peeking penalty calculator |
| tab4 | Strategic Matrix | CR vs AOV scatter with quadrant labels; `plot_strategic_matrix()` |
| tab5 | Product Metrics | APO and APU bar charts; `plot_multivariant_bar()` |
| tab6 | Revenue Charts | RPV and AOV bar charts; `plot_multivariant_bar()` |
| tab7 | CR Comparison | CR bar chart; `plot_multivariant_bar()` |
| tab8 | Bayesian PDFs | Overlapping Beta posteriors for all groups; `plot_bayesian_pdfs()` |
| tab9 | Bootstrap | Bootstrap CI histogram for Control vs `best_m`; stores results in session_state |
| tab10 | Box Plot | Box plot of bootstrap distributions; reads from session_state set by tab9 |

**tab9 / tab10 dependency:** tab10 reads `_boot_samples_c` and `_boot_samples_v` from `st.session_state`. These are written by tab9 when it renders. If the user opens tab10 before tab9, it shows an info message. This is intentional — do not remove it.

---

## Multi-Variant Mode vs Classic A/B

The app has one code path, not two. Classic A/B is just the multi-variant engine with `num_variations=1`.

When `num_variations == 1`:
- Omnibus chi-square equals the z-test (two groups)
- `multipletests` with one p-value returns it unchanged
- The omnibus banner is hidden
- The correction selector is not shown in the sidebar

When `num_variations > 1`:
- Omnibus banner is shown with global p-value
- Correction selector appears in sidebar
- KPI table shows all groups
- All charts render N bars / N dots

---

## Revenue Significance: Key Limitations

- **Only Control vs best variation** is tested, not all pairwise revenue comparisons
- Uses **reconstructed distributions** (log-normal from aggregates) — not raw order data
- Fixed random seed (42) makes results reproducible across rerenders
- AOV arrays contain only converters; RPV arrays contain all visitors (zeros for non-converters)
- Treat results as directional signals, not precise frequentist p-values

---

## AI Prompt Structure

The AI analysis prompt requires exactly these 9 sections in order:

1. Executive Summary
2. Trade-off Analysis
3. Risk Assessment
4. Visual Insights > Strategic Matrix
5. Visual Insights > Product Metrics
6. Visual Insights > Revenue Charts
7. Visual Insights > CR Comparison
8. Visual Insights > Bayesian Posterior
9. Visual Insights > Bootstrap Distribution
10. Visual Insights > Box Plot
11. Recommendation

The prompt instructs the model **not** to add a title or restate the hypothesis as a heading. Doing so caused a redundant header bug that was previously fixed. Do not remove this instruction.

Both OpenAI (GPT-4o) and DeepSeek (R1) use the same OpenAI-compatible client. The only differences are `base_url` and `model_name`.

---

## Plotting Conventions

All plot functions call `plt.close(fig)` after `st.pyplot(fig)` to prevent memory leaks across rerenders.

`matplotlib.use("Agg")` is set at module level to prevent threading crashes in Streamlit's server environment. Do not remove it.

Multi-variant bar charts (`plot_multivariant_bar`) colour bars using `GROUP_COLORS` — blue for Control, then green/orange/purple for variations. Bars that are below the control value are coloured red (`#d62728`) regardless of group index.

---

## Editing Rules

1. **Always read the section you are editing before changing it.** The file is 1,040 lines and the view tool truncates. Use `view_range` to read specific sections.

2. **Do not reorder top-level sections.** The sidebar must render before the computation block, which must run before the dashboard and tabs.

3. **All new statistical functions go between `perform_srm_test` and `run_multivariate_analysis`.** Keep the function block contiguous.

4. **New sidebar widgets go inside the "Data Inputs" expander or below `days_run`, before the computation block.** Never add widget calls after the computation block.

5. **Adding a new tab:** Add the variable to the `st.tabs([...])` call, increment tab count, and add a `with tabN:` block at the bottom. Tab index must match declaration order.

6. **State defaults:** Any new persistent input must be added to `initialize_state()` defaults AND to `SAVE_KEYS` if it should be saved in JSON snapshots.

7. **Verify syntax** after every edit: `python3 -m py_compile ab_test_analyzer.py`

8. **Do not use `st.experimental_*` APIs** — they are removed in recent Streamlit versions.

---

## Known Design Decisions & Past Bug Fixes

| Decision | Reason |
|---|---|
| `chisquare` for SRM, not `chi2_contingency` | `chi2_contingency` tests independence (wrong); `chisquare` tests goodness-of-fit vs expected allocation (correct) |
| `matplotlib.use("Agg")` set at module level | Prevents threading crashes in Streamlit's server |
| `plt.close(fig)` after every `st.pyplot()` | Prevents memory accumulation across rerenders |
| Bootstrap stores results in `session_state` | tab10 needs tab9's data but tabs render independently; state bridge is the safe pattern |
| Revenue sig only tests Control vs best | All-pairwise revenue tests are expensive and directional signals are sufficient at this stage |
| Winner = highest CR uplift among significant | RPV could theoretically be the tiebreaker, but CR is the primary hypothesis metric |
| No AI report title / no hypothesis header | Previous version re-rendered the hypothesis as a header, creating a redundant heading above the AI output |
| Omnibus banner only shown for multi-variant | In classic A/B, the omnibus and z-test are identical — showing both would be confusing |

---

## Planned Enhancements (Not Yet Implemented)

These were agreed upon in previous sessions but not yet built. Check before starting any of them.

1. ~~Revenue significance testing — Mann-Whitney U / log t-test on AOV and RPV~~ ✅ Done
2. ~~Multi-variant support — A/B/n with multiple comparison correction~~ ✅ Done
3. ~~MDE visualisation — power curves showing effect size vs sample size~~ ✅ Done
4. Segment breakdown explorer — per-segment CR/RPV with filters
5. CUPED variance reduction — covariate adjustment using pre-experiment metric
6. PDF export — downloadable report from the full analysis
7. Guardrail metrics section — secondary metrics with their own significance tests
8. Test history log — persistent record of past experiments
9. Smarter duration warnings — business cycle checks, day-of-week bias detection

---

## Dependencies

```
streamlit
numpy
pandas
scipy          # beta, chisquare, mannwhitneyu, chi2_contingency
statsmodels    # proportions_ztest, proportion_confint, proportion_effectsize,
               # NormalIndPower, TTestIndPower, multipletests
matplotlib
openai         # used for both OpenAI and DeepSeek via base_url override
```

No pinned versions. The app was built and tested against current stable releases as of early 2026.
