# Audit: "PCA of Indian Equity Returns" — Report, Slides, and Notebooks

**Scope reviewed:** `pca_project_report.pdf`, `pca_final.pdf`, `stat4_pca_project_final.ipynb` (the notebook submitted with the report), `pca_project_notebook.ipynb` (your extended, ongoing research notebook), and Dr. De's written feedback (`skd_20_04_26.txt`).

**How to read this document:** Section 1 answers Dr. De's four comments directly, with evidence pulled from your own code and outputs — not just abstract advice. Section 2 covers problems I found that he didn't mention but that matter for the same reasons he gave. Section 3 is a prioritized fix list. Section 4 is corrected language you can reuse in a v2 report/abstract.

---

## 1. Dr. De's four comments, answered with your own evidence

### 1.1 "Use reliable sources only"

Your reference list (`README.md` links) is a mix of three tiers:

- **Tier 1 (keep):** peer-reviewed papers and course notes — Fama & French (1993), Connor & Korajczyk (1988), Chow (1960), Marchenko & Pastur (1967), the MIT 18.S096 lecture notes, Avellaneda's NYU lecture notes, the arXiv papers.
- **Tier 2 (use with attribution, not as authority):** Stack Exchange/Quant SE threads, QuantInsti/IBKR/Macrosynergy blog posts. These are fine for *intuition and implementation tricks*, never as the justification for a methodological choice in a report.
- **Tier 3 (drop entirely):** the NJ Factorbook / NJ Mutual Fund pages, Medium posts, LinkedIn Pulse posts, Reddit threads, Bogleheads forum, YouTube videos. These are exactly the "Stanford-guy-blog" category Dr. De is warning you away from — several are literally AMC (asset management company) content-marketing pages designed to sell mutual funds, not neutral methodological references. See `FACTOR_ANALYSIS_KNOWLEDGE_BASE.md` for a full triage of your link list with substitutes.

Action: in any resubmission, your bibliography should cite only Tier 1 sources for methodology claims (what MP theory says, what a Chow test assumes, what "factor" means). Tier 2/3 material can inform your engineering choices but shouldn't appear as a citation next to a claim.

### 1.2 "Keep the method simple first — know what's really happening"

Two concrete places where the notebook is more complicated than it is *validated*, which is the opposite of what Dr. De is asking for:

**(a) The Marchenko–Pastur criterion is claimed but never computed.**
Both notebooks describe MP as "the primary criterion" for choosing the number of components (`stat4_pca_project_final.ipynb`, cell 5: *"Marchenko-Pastur Law | Retain eigenvalues λ_k > λ+^MP ... The primary criterion"*). I searched the entire executed notebook for where `λ+^MP` is actually calculated and compared against an eigenvalue — **it never happens**. Every actual PCA call in the notebook hardcodes the component count (`n_components=10`, `n_components=5`) instead of deriving it from a threshold. This is a "say one thing, do another" gap that would be very easy for an examiner to catch by asking "show me the line where you compute the MP bound" — which may be close to what happened in your presentation.

Related: the "three criteria must all agree" framework (scree / 70% cumulative EVR / MP) is internally inconsistent for your own data. Your top 10 PCs on the monthly cross-sectionally-standardized panel explain only ~24–27% cumulative variance (Table 2 in the report). To reach a 70% cumulative-EVR threshold you would need on the order of 60+ of your 118 components — i.e., almost no compression at all. So the "70% EVR" criterion, as stated, could never plausibly agree with a 3–5 component MP-based choice on this dataset. Either the criterion needs to be dropped/rephrased for the cross-sectionally-standardized case, or you need to state explicitly that you are *not* using cumulative EVR as a binding criterion here and explain why.

**(b) The risk-free-rate subtraction is a no-op, and that's worth knowing.**
`RF_DAILY = 0.065/365` (and the monthly equivalent) is subtracted from every stock's return before cross-sectional standardization (`cross_section_standardize`) and before the rolling within-window standardization. Both of those standardizations subtract the mean and divide by the standard deviation *of the same cross-section/window*. Subtracting a constant that is identical for every stock at a given date does not change either the cross-sectional mean-centering or the within-window centering — it cancels out exactly. In other words, **the "excess return" step has zero numerical effect on any PCA result in this project**, because the flat 6.5% risk-free rate is a constant shift, not a cross-sectionally or temporally varying one. This isn't a bug (nothing breaks), but it is exactly the kind of thing Dr. De means by "be able to explain what is really happening" — the code has a step that looks like it matters and doesn't, and that's worth being able to say out loud rather than have him find it.

**(c) Ledoit–Wolf shrinkage changes what "explained variance" means, without a note on it.**
You correctly use Ledoit–Wolf shrinkage instead of the raw sample covariance in `run_pca()` (good instinct for T≈24, D≈100+). But once you shrink the covariance matrix, the reported eigenvalues/EVR are eigenvalues of the *shrunk* matrix, not the sample covariance — they are pulled toward the shrinkage target (a scaled identity matrix) by an amount that depends on the estimated shrinkage intensity. This makes cross-window EVR comparisons subtly apples-to-oranges if the shrinkage intensity itself drifts over time (which it will, since it depends on window sample size, dimensionality, and cross-sectional dispersion — all constant here, but it depends on estimated data moments each window). Worth one sentence in the paper acknowledging this, and — if you want to check it's not distorting your headline result — rerun the rolling PC1-series without shrinkage and see if the pre/post-COVID gap survives.

### 1.3 "Be careful with PCA interpretation — PCA is not factor analysis"

Your report is generally careful about this in the prose *definitions* (the abstract correctly says "the extracted factors correspond to style and sector-based co-movements rather than the broad market beta," and you cite Connor & Korajczyk correctly). But the report and notebook titles and section headers ("Factor Discovery," "the factor structure," "latent factors") use "factor" and "principal component" interchangeably throughout, without ever stating the actual mathematical distinction. Here it is, precisely, so you can put one paragraph in the report that will preempt this exact question:

> **PCA (as you ran it) and classical Factor Analysis solve superficially similar but different problems.** Both decompose a return matrix into `common part + idiosyncratic part`. Factor Analysis (via MLE/EM) allows the idiosyncratic covariance Σ_ε to be an arbitrary diagonal matrix — each stock can have its own idiosyncratic variance. Standard PCA is equivalent to *probabilistic PCA*, which is the special case where every stock is assumed to have the **same** idiosyncratic variance σ² (Tipping & Bishop, 1999). PCA doesn't need this assumption to *run* — the algorithm does the same eigendecomposition regardless — but the assumption is implicit in treating PCA's output as if it were estimating the same object Factor Analysis estimates. The reason PCA is still defensible for equities specifically is a separate, asymptotic argument due to Connor & Korajczyk (1986, 1988): as the number of assets D → ∞, if the true model is an *approximate* factor model (idiosyncratic variances bounded, common-factor eigenvalues growing with D), PCA's components converge to the true factor space even without the equal-variance assumption holding exactly. That asymptotic argument is *why* you're allowed to use PCA here, and your report should say so explicitly rather than just citing Connor & Korajczyk in a sentence and moving on.

Practical add-on for the repo: run an actual MLE/EM Factor Analysis (`sklearn.decomposition.FactorAnalysis`, or the EM algorithm) on the same standardized panel with the same K, and report how different the loadings are from PCA's. If they're similar, that's your empirical justification that the equal-idiosyncratic-variance assumption is not badly violated here. If they differ a lot, that itself is a finding worth reporting.

### 1.4 "Do not overstate structural change — know the test's assumptions and limits"

This is the most serious issue in the project, and your own extended notebook already contains the evidence that undercuts the headline claim in the submitted report. Walking through it:

**What the submitted report claims:** "A Chow test (F = 22.93, p < 0.001) rejects stability... indicating a permanent increase in return synchronisation," and the recommendations section tells policymakers and fund managers to act on this as a "permanent structural break."

**Problem 1 — the Chow test's own assumptions are violated by construction.** The series being tested (`pc1_var_series`, the rolling PC1 variance) is built from a **24-month window stepped forward one month at a time** (`step=1`). Any two adjacent observations share 23 of their 24 underlying months. This overlapping-window construction manufactures strong serial autocorrelation in the series *by design*, regardless of whether the true underlying process has any autocorrelation at all — a classic pitfall with overlapping-window statistics (the same issue that Richardson & Stock (1989) raised for overlapping-return regressions). The classical Chow test's F-distribution assumes independent, homoskedastic residuals. With effectively ~112/24 ≈ 4–5 independent blocks of information rather than 112 independent observations, the true degrees of freedom are far smaller than the (2, 108) used in your code (`chow_test()`, `df2 = T - 2*k`), and the p-value is mechanically overstated (too significant). This is not a coding bug — `chow_test()` is implemented correctly for the *classical* Chow test — the problem is applying the classical (i.i.d.-residual) Chow test to a manufactured-autocorrelated series without a Newey–West/HAC correction, or without first constructing the test on non-overlapping windows.

**Problem 2 — your own more careful analysis already found the opposite result, and it didn't make it into the final report.** `pca_project_notebook.ipynb` (cell 80) runs a **permutation test** on the same pre/post split — a test that doesn't rely on the classical F-distribution or independence assumptions at all — and gets **p = 0.1695: not significant.** The same cell reports **bootstrap confidence intervals for PC1 that overlap substantially** (pre: [14.87, 33.85], post: [18.57, 72.98]) and explicitly concludes: *"The post-break CI is very wide — indicating instability rather than a sharp level shift."* This is exactly Dr. De's point, made by your own more rigorous test. The submitted report and notebook report only the Chow test and drop the permutation test and bootstrap CIs — which, intentionally or not, is reporting the significant test and omitting the non-significant one on the same question. That's the kind of selective reporting that makes a structural-break claim "too strong or unsupported," in his words.

**Problem 3 — even a well-specified test would only tell you "correlations rose," which is not new or necessarily permanent.** It is well documented in the finance literature (Longin & Solnik, 2001; Ang & Chen, 2002) that equity correlations rise during market downturns generally, not uniquely after COVID — this is sometimes called correlation asymmetry or "correlation breakdown." Your post-COVID window (2020–2026) also contains the 2022 global rate-hike shock, so "elevated co-movement since COVID" is confounded with "elevated co-movement during a multi-shock six-year stretch that happens to start after COVID." A defensible structural-break claim needs either (a) a formally identified break date (see §2.3 below) rather than one fixed a priori at March 2020, or (b) an explicit acknowledgment that you cannot separate a COVID-specific break from a general rise in market turbulence over 2020–2026.

**Bottom line for a resubmission:** report the Chow test, the permutation test, and the bootstrap CIs together, say plainly that they disagree, and downgrade the conclusion from *"a permanent structural break occurred"* to *"co-movement was higher on average post-2020, but non-parametric tests do not confidently distinguish this from sampling variability given the short post-break sample; the Procrustes evidence on loading composition is the strongest and most robust finding."* (The Procrustes result is in fact your best evidence — see §2.3.)

---

## 2. Additional issues (not raised by Dr. De, but the same class of problem)

### 2.1 Reproducibility: the same headline number changes across cells/runs

Three different runs of "what was PC1 variance explained, pre- vs. post-COVID" appear in your two notebooks, and they don't agree:

| Source | Pre-COVID PC1 var. | Post-COVID PC1 var. | Chow F |
|---|---|---|---|
| `stat4_pca_project_final.ipynb`, cell 43/45 (submitted, matches the report) | 15.9% | 19.8% | 22.93 |
| `stat4_pca_project_final.ipynb`, cell 50 (markdown interpretation, same notebook) | ~20% | ~25% | **24.57** |
| `pca_project_notebook.ipynb`, cell 80 (extended notebook) | 25.5% | 39.0% | 24.57 |
| `pca_project_notebook.ipynb`, cell 165 (Section 6 summary, same notebook) | 16.5% | 21.0% | — |

Four different pairs of numbers for what should be one fact. This tells me the pipeline was rerun multiple times (different universe sizes — 106 vs. 117 vs. 118 stocks — different standardization settings, or a fresh `yfinance` pull that silently changed the underlying prices) without the write-ups being regenerated from the final run. This is worth fixing before anything else: **pick one canonical, version-controlled data snapshot and one configuration, regenerate every number in the report from that single run, and only then write the interpretation.** Right now, no number in the report can be fully trusted at face value without being traced back to a specific, current run.

### 2.2 Stale interpretation text that contradicts its own preceding code output

Two markdown cells in `stat4_pca_project_final.ipynb` describe results that don't match the code cell immediately above them:

- **Cell 39** claims PC1 "explains approximately 28–35%" of daily variance and is "unambiguously... the market factor" because "over 85% of stocks carry positive PC1 loadings," naming HDFC Bank/ICICI Bank/Reliance as top loaders. The code in cells 34 and 38 immediately above it printed **4.33%** daily variance and **52%** positive loadings, with top stocks ELECON/INOXWIND/GOKEX/CANBK/TANLA — and the code's own auto-generated label for this is *"STYLE FACTOR,"* not a market factor. Cell 39 appears to be leftover text from an earlier, different run of the pipeline (possibly one that didn't cross-sectionally standardize, which is exactly the transformation that removes the market component and would explain the "over 85% positive, market factor" language).
- **Cell 50** has the same problem for the COVID-break numbers, as shown in the table above.

This matters practically: if you presented from the notebook (rather than only the static report PDF), Dr. De may have been looking at cell 39/50's text while your slides showed the report's correct numbers — which alone could produce exactly the "doesn't make sense to him" reaction. Before any future presentation, delete or regenerate every markdown interpretation cell so it's mechanically tied to the cell above it, ideally by having the notebook compute and f-string the numbers into the markdown rather than typing them by hand.

### 2.3 The sample is right at the edge of "no genuine multi-factor structure exists"

For the **monthly** full-sample PCA (N=135 months, D=118 stocks), N/D ≈ 1.14 — this is deep inside the regime where Marchenko–Pastur theory says almost the *entire* eigenvalue spectrum is consistent with pure noise (the MP bound only shrinks toward 0 as N/D → ∞; at N/D near 1, the bound is very wide). This doesn't mean PC1–PC4 are meaningless, but it does mean the paper is on much shakier ground quantitatively interpreting PC3/PC4 (2–3% variance each) than it currently reads. The **daily** panel (N/D ≈ 23.6) is much better-conditioned and is where your more defensible numbers come from — consider leading with daily-frequency PCA for factor identification and using monthly only for the economically-interpretable, lower-frequency story, with an explicit caveat on the monthly N/D ratio.

Your rolling-window analysis is worse on this axis: N=24 months, D≈106–118 stocks gives N/D ≈ 0.2–0.23 (D/N ≈ 4.4–4.9). At that ratio the MP upper bound is roughly `(1+√4.4)² ≈ 9.6` in units of the noise variance — meaning an eigenvalue would need to be nearly 10× the average noise eigenvalue to be distinguishable from noise. This is likely fine for PC1 (which is large) but is a real reason to distrust PC2–PC5 in the rolling analysis specifically, which is one more reason the Ledoit–Wolf shrinkage matters and should be reported alongside an explicit MP bound number for the window size actually used.

**What *is* robust:** the Procrustes disparity result (δ ≈ 0.78 between pre/post full-sample loadings) doesn't depend on the Chow test's assumptions at all — it's a geometric comparison of two independently-estimated loading matrices. This is your strongest, least fragile finding, and it should be promoted to the headline result instead of the Chow test.

### 2.4 PC "economic labels" are unvalidated, and you already know this

Both notebooks correctly caveat that PC-to-economic-factor labels ("domestic cyclical vs. defensive growth," "technology vs. commodities") are post-hoc narrative rather than statistical fact. Good practice — but the natural next step, which the Fan–Liao–Wang "Projected PCA" framework and the older Connor–Linton (2007) / Connor–Hagmann–Linton (2012) semiparametric approach both provide, is to *formally test* whether a stock characteristic (market cap, sector, book-to-market, past return) explains the loadings, rather than reading them off a sorted list of tickers. Concretely: regress each PC's loading vector cross-sectionally on log market cap, sector dummies, and a momentum proxy; a high R² supports your label, a low one means the "style/sector" story is decoration. This is exactly the kind of "can you justify the distinction properly" evidence Dr. De is asking for in §1.3.

### 2.5 Data quality: yfinance for NSE-listed corporate actions

118 of 145 candidate tickers passed your audit, but the audit criterion allows up to **4** single-day returns beyond ±50% per stock before exclusion (Table 1: max "Extreme rets" = 4). A single-day 50%+ move in a large/mid-cap Indian stock is almost always a **data artifact** (an unadjusted bonus issue, stock split, or demerger that `yfinance`'s auto-adjustment missed) rather than a genuine return, and even one such point can materially distort a covariance estimate built from only 24–135 observations. Before the next run, spot-check the specific dates flagged as "extreme" against NSE corporate-action records for the affected tickers, and consider tightening the threshold to 0–1 extreme observations rather than 5.

### 2.6 Known bugs your own extended notebook already flagged (prioritize these)

`pca_project_notebook.ipynb`, cell 166 ("Section 7 — Recommendations") already contains a self-review with five items your own process identified as needing fixes before further use:

1. `AlphaEngine101` constructor keyword bug (`open=` should be `open_=`) — breaks a fresh run of the alpha section.
2. Eigenvector sign-flip is not corrected between consecutive expanding-window fits in the walk-forward meta-alpha construction — this is likely inflating apparent instability/hurting the walk-forward Sharpe (noted potential improvement from −0.65 to +0.4–0.6 once fixed).
3. The 15-row robustness table for sector rotation has 8 duplicated rows, making the "tested across configurations" claim look more thorough than it is.
4. The Chow-test note needs the trend-vs-mean-break clarification (partially addresses §1.4 above, but doesn't address the autocorrelation problem).
5. `MetaAlpha_3` is mislabeled "STABLE" when its sign-consistency check should mark it "WEAK/WRONG."

These are worth fixing first, before adding anything new, because they affect the credibility of everything built on top of them (the backtested Sharpe ratios in particular).

---

## 3. Prioritized action list

**Before anything else (data integrity):**
1. Freeze one data snapshot (save the raw prices to disk/version control) and one configuration. Regenerate every number in the report from that single run. Note the snapshot date and `yfinance` version used.
2. Fix the five self-flagged bugs in `pca_project_notebook.ipynb` cell 166.
3. Re-derive every markdown interpretation cell from the code output programmatically (f-strings), not by hand, so they can't drift out of sync again.

**Statistical rigor (addresses Dr. De's feedback directly):**
4. Actually compute the MP bound (`λ_+^MP = σ²(1+√(D/N))²`) for both the full-sample and rolling-window panels, plot it against the scree plot, and use it (or explicitly say why you're overriding it) — don't just narrate it.
5. Report the Chow test, the permutation test, and the bootstrap CIs for the structural-break claim together, and soften the conclusion to match what all three jointly support (see §1.4).
6. Add a Bai–Ng (2002) or Ahn–Horenstein (2013) eigenvalue-ratio-based component count as a cross-check against the ad hoc `n_components=5/10`.
7. Add one paragraph distinguishing PCA from Factor Analysis using the equal- vs. heteroskedastic-idiosyncratic-variance criterion (§1.3), and optionally run `sklearn.decomposition.FactorAnalysis` on the same panel as a robustness check.
8. Run the rolling PC1 series without Ledoit–Wolf shrinkage as a robustness check on the headline pre/post-COVID gap.

**Polish before any resubmission or re-presentation:**
9. Replace all Tier-2/3 citations (see `FACTOR_ANALYSIS_KNOWLEDGE_BASE.md`) with Tier-1 equivalents.
10. Rewrite the abstract/conclusion using the corrected language in §4 below.

---

## 4. Corrected language you can reuse

**Original abstract claim:**
> "A 24-month rolling PCA shows that PC1 explained variance rose from a pre-COVID mean of 15.9% to a post-COVID mean of 19.8%, indicating a permanent increase in return synchronisation. A Chow test (F = 22.93, p < 0.001) rejects stability of the linear variance trend at the break."

**Suggested replacement:**
> "A 24-month rolling PCA shows PC1 explained variance was higher on average post-March-2020 (19.8%) than before (15.9%). A classical Chow test on this series is significant (F = 22.93, p < 0.001), but the rolling series is constructed from heavily overlapping windows, which mechanically inflates apparent significance; a non-parametric permutation test on the same sub-period means is not significant (p = 0.17), and bootstrap confidence intervals for pre- and post-break PC1 variance overlap substantially. We therefore do not claim a statistically confirmed permanent break in co-movement *level*. The strongest and most robust evidence of change is compositional: an orthogonal Procrustes alignment of the pre- and post-COVID loading matrices gives a disparity of 0.78 (0 = identical, 1 = orthogonal), indicating that *which* stocks drive systematic co-movement shifted substantially around the COVID shock, even where the overall *amount* of co-movement is harder to pin down statistically. This is also consistent with a broader empirical literature (Longin & Solnik, 2001; Ang & Chen, 2002) showing equity correlations rise in market downturns generally, so some of the observed change may reflect general post-2020 market turbulence rather than a COVID-specific, permanent regime shift."

This version keeps every number you already have, adds the tests you already ran but didn't report, and downgrades the causal/permanence language to what the combined evidence actually supports — which is precisely what Dr. De is asking for.
