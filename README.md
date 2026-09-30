# Market Factor Analysis — PCA & Factor Analysis of Indian Equities

> **Applying Principal Component Analysis and Factor Analysis to NSE/BSE equity returns to identify latent market factors, test structural change, and build a foundation for systematic factor investing.**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Setup & Reproducibility](#setup--reproducibility)
- [Key Results (v2.0)](#key-results-v20)
- [Roadmap — Future Work](#roadmap--future-work)
- [Reference Tiers](#reference-tiers)
- [Citation](#citation)

---

## Project Overview

This project is a rigorous research notebook studying the **factor structure of Indian equity returns (NSE/BSE, 2015–2026)** using:

- **Principal Component Analysis (PCA)** with Ledoit-Wolf covariance shrinkage
- **MLE Factor Analysis** (and PCA vs FA comparison via Procrustes alignment)
- **Marchenko-Pastur Random Matrix Theory** for principled component selection
- **Structural break testing** around COVID-19 (March 2020) via a full battery:
  - Classical Chow (1960) test with documented autocorrelation caveats
  - Non-parametric permutation test (assumption-free)
  - Circular block bootstrap CIs — Politis & Romano (1994)
  - HAC-corrected regression — Newey & West (1987), lag=23
  - Orthogonal Procrustes alignment of pre/post loading matrices

The project was developed in response to a formal academic peer review and directly addresses each reviewer comment with code, math, and honest reporting of all test results — including tests that disagree.

**Key philosophy:** where tests agree, the result is stated with confidence. Where they disagree (Chow vs permutation test), both are reported and the discrepancy explained.

---

## Repository Structure

```
market-factor-analysis/
│
├── pca_project_notebook.ipynb        <- Main research notebook (v2.0) — START HERE
├── pca_project_notebook_old.ipynb    <- Archived v1 (reference only; contains known bugs)
│
├── data_cache_v2/                    <- Frozen data + output figures
│   ├── prices_raw.parquet            <- Static price snapshot (reproducibility anchor)
│   ├── universe_log.json             <- Download metadata (date, yfinance version)
│   ├── canonical_numbers.json        <- Single source of truth for all reported numbers
│   ├── scree_mp_bounds.png           <- Scree plot with Marchenko-Pastur bounds
│   ├── loading_heatmap.png           <- PCA loading heatmap (all K components)
│   ├── pca_vs_fa_scatter.png         <- PCA vs FA loading comparison
│   ├── permutation_test.png          <- Permutation distribution for structural break
│   ├── rolling_pc1_evr.png           <- Rolling 24-month PC1 EVR time series
│   └── procrustes_scatter.png        <- Pre vs post-COVID loading alignment
│
├── research papers/
│   ├── links.txt                     <- Curated reading list (74 sources, tiered)
│   ├── pca_theory.md                 <- Mathematical notes on PCA / FA / RMT
│   ├── factor_investing.md           <- Notes on factor investing theory
│   ├── Fama-French_JFE93.pdf         <- Fama & French (1993)
│   ├── fama2015.pdf                  <- Fama & French (2015)
│   ├── The-Arbitrage-Theory-of-Capital-Asset-Pricing.pdf  <- Ross (1976)
│   ├── Perold-CapitalAssetPricing-2004.pdf
│   ├── 101_ALPHA.pdf                 <- Kakushadze (2016) — 101 Formulaic Alphas
│   ├── 151-Trading-Strategies.pdf    <- Jansen (2020)
│   └── DL-40-Factor.pdf              <- Luan (2025) — 40 Behavioral Factors
│
├── PROJECT_AUDIT.md                  <- Peer review comments + detailed responses
├── FACTOR_ANALYSIS_KNOWLEDGE_BASE.md <- Curated knowledge base on factor investing
└── README.md                         <- This file
```

---

## Methodology

### Data
- **Universe:** 100 NSE/BSE stocks (from ~118 candidates), Jan 2015 – Apr 2026
- **Source:** Yahoo Finance via `yfinance`; frozen to `data_cache_v2/prices_raw.parquet`
- **Quality gates:** coverage >= 85%; corporate-action extreme return audit (|r| > 50%)
- **Known limitation:** survivorship bias — delisted/merged/recategorised firms excluded

### Returns
- Daily log-returns: `r_{i,t} = log(P_{i,t} / P_{i,t-1})`
- Monthly: sum of daily log-returns within each calendar month
- Cross-sectional standardization at each date: `r_tilde = (r - mean_cross) / std_cross`
  - Note: with a flat daily RF rate subtracted uniformly, XS-standardization is unchanged; RF subtraction is a verified no-op (Section 2.5)

### PCA Pipeline
1. **Covariance:** Ledoit-Wolf (2004) analytical shrinkage; raw covariance run in parallel for robustness
2. **Component selection:** MP upper bound lambda+ = sigma^2 * (1 + sqrt(q))^2; Bai-Ng (2002) IC criteria; Ahn-Horenstein (2013) eigenvalue ratio
3. **Loadings:** `numpy.linalg.eigh` (symmetric, stable); sign convention: max absolute loading is positive
4. **Validation:** PCA loadings regressed on cap-tier + sector dummies (Section 3.6); compared to MLE FA via Procrustes alignment (Section 3.5)

### Structural Break Tests (Section 4)

| Test | Accounts for autocorrelation? | Role |
|------|------|------|
| Classical Chow | No (documented) | Baseline; unreliable on overlapping-window series |
| Permutation test | Yes (assumption-free) | Primary inference tool |
| Circular block bootstrap CI | Yes (block_len=24) | Honest uncertainty quantification |
| HAC regression (NW lag=23) | Yes | Most honest parametric version |
| Non-overlapping block t-test | Yes | ~5 independent obs — low power, honest |
| Procrustes alignment | Yes (two independent estimates) | Most robust finding |

---

## Setup & Reproducibility

### Requirements

```bash
pip install numpy pandas scipy statsmodels scikit-learn matplotlib yfinance pyarrow
```

### Running the notebook

```bash
jupyter notebook pca_project_notebook.ipynb
```

Run cells top-to-bottom. Cell 0-A checks for the cached parquet — on first run it downloads from yfinance and caches; on all subsequent runs it reads the frozen file. Every reported number traces back to `canonical_numbers.json`.

---

## Key Results (v2.0)

> Run the notebook to populate exact numbers. Structural template:

- Daily panel: **K significant components** above the Marchenko-Pastur upper bound (N/D = 27.8)
- PC1 EVR (daily, LW shrinkage): broad market co-movement factor; fraction positive loadings near 1
- PCA vs FA: high Pearson r for PC1 loadings — empirical support for approximate factor model
- Pre-COVID mean PC1 EVR: ~3.97% | Post-COVID: ~4.74%
- Permutation test p-value: see Cell 4-E output
- Block bootstrap 95% CIs: see Cell 4-F output
- Procrustes disparity delta: see Cell 4-H output

**Honest conclusion:** Post-COVID average PC1 EVR is higher on average. A classical Chow test is significant but unreliable (manufactured autocorrelation, DW near 0). Non-parametric tests give a more honest picture. The strongest, most assumption-free finding is Procrustes compositional change: the stocks driving systematic co-movement shifted around the COVID shock.

---

## Roadmap — Future Work

### Phase 2 — Factor Validation & Economic Labeling
- [ ] Add fundamentals (P/B, ROE, debt/equity) via Screener.in
- [ ] Fama-MacBeth (1973) two-pass regression for factor risk premia
- [ ] Compare PC factors to Fama-French 3F and 5F on Indian data
- [ ] Macro overlay: FII flows, USD/INR, WTI oil, RBI rate decisions

### Phase 3 — Rolling Stability & Regime Detection
- [ ] Andrews (1993) Sup-F test for unknown break date
- [ ] Bai-Perron (1998) multiple break-point estimation
- [ ] Factor mimicking portfolio construction
- [ ] Factor stability analysis across detected regimes

### Phase 4 — Applications
- [ ] Factor covariance-based mean-variance optimization
- [ ] Equal-risk-contribution and maximum-diversification portfolios
- [ ] Walk-forward backtest with explicit NSE transaction costs
- [ ] PCA on 101 Formulaic Alphas (Kakushadze 2016) → meta-alpha
- [ ] PCA on 40 Behavioral Alphas (Luan 2025) → behavioral meta-factor

### Phase 5 — Advanced Techniques
- [ ] Sparse PCA — Zou, Hastie & Tibshirani (2006)
- [ ] POET — Fan, Liao & Mincheva (2013)
- [ ] Dynamic Factor Models — Stock & Watson (2002)
- [ ] Independent Component Analysis for non-Gaussian factors
- [ ] Deep learning factor models — Kelly, Pruitt & Su (2019); Gu, Kelly & Xiu (2020)

---

## Reference Tiers

**Tier 1 — peer-reviewed (cited in notebook for all methodology claims):**

| Paper | Role |
|-------|------|
| Ahn & Horenstein (2013) *Econometrica* | Component selection — eigenvalue ratio |
| Andrews (1993) *Econometrica* | Sup-F structural break test |
| Ang & Chen (2002) *J. Financial Economics* | Correlation asymmetry in downturns |
| Bai & Ng (2002) *Econometrica* | Component selection — IC criteria |
| Chow (1960) *Econometrica* | Structural break test |
| Connor & Korajczyk (1988) *J. Finance* | PCA in approximate factor models |
| Fama & French (1993) *J. Financial Economics* | Three-factor model |
| Fama & French (2015) *J. Financial Economics* | Five-factor model |
| Harvey, Liu & Zhu (2016) *Review of Financial Studies* | Factor zoo / multiple testing |
| Laloux et al. (1999) *Phys. Rev. Lett.* | MP law applied to equity matrices |
| Ledoit & Wolf (2004) *J. Multivariate Analysis* | Analytical shrinkage estimator |
| Longin & Solnik (2001) *J. Finance* | Extreme correlations in downturns |
| Marchenko & Pastur (1967) *Math. USSR-Sbornik* | Random matrix eigenvalue distribution |
| Newey & West (1987) *Econometrica* | HAC standard errors |
| Plerou et al. (1999) *Phys. Rev. Lett.* | Cross-correlations in equity returns |
| Politis & Romano (1994) *JASA* | Circular block bootstrap |
| Ross (1976) *J. Economic Theory* | Arbitrage Pricing Theory |
| Sharpe (1964) *J. Finance* | CAPM |
| Tipping & Bishop (1999) *J. Royal Statistical Society B* | Probabilistic PCA vs FA |

**Tier 2 — practitioner resources (implementation intuition; not cited for methodology):**
Avellaneda NYU lecture notes; QuantInsti blog; IBKR Quant; Macrosynergy; MIT 18.S096 notes.

**Tier 3 — excluded from citations:**
NJ Factorbook / NJ MF pages (AMC marketing), Reddit, Medium, YouTube, LinkedIn Pulse, Bogleheads.
Logged in `research papers/links.txt` for personal reading reference only.

---

## Citation

```
Singh, A. (2026). Market Factor Analysis: PCA & Factor Analysis of Indian Equity Returns.
GitHub: https://github.com/amansingh2116/market-factor-analysis
```

---

*Notebook v2.0 | Last updated: September 2026 | Python 3.12*
