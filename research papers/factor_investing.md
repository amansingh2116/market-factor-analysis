# Factor Analysis, PCA, and Factor Investing in Quantitative Finance

## A Comprehensive Guide — From Mathematical Foundations to Portfolio Implementation

---

> **How to read this document:** This guide is structured as a learning journey. Part I builds the statistical machinery (Factor Analysis, PCA). Part II connects it to finance theory (CAPM, APT, Fama-French). Part III covers practical applications (factor regressions, portfolio construction, India). Each section references the others — you can read linearly or jump to what you need.

---

# Table of Contents

1. [Why Dimensionality Reduction? The Core Problem](#1-why-dimensionality-reduction-the-core-problem)
2. [Factor Analysis — The Probabilistic Foundation](#2-factor-analysis--the-probabilistic-foundation)
3. [Probabilistic PCA — The Bridge to Classical PCA](#3-probabilistic-pca--the-bridge-to-classical-pca)
4. [Principal Component Analysis (PCA)](#4-principal-component-analysis-pca)
5. [PCA — Example and Interpretation](#5-pca--example-and-interpretation)
6. [Exploratory Factor Analysis and Its Limitations](#6-exploratory-factor-analysis-and-its-limitations)
7. [Factors in Investing — History and Theory](#7-factors-in-investing--history-and-theory)
8. [The Economic Mechanisms Behind Factors](#8-the-economic-mechanisms-behind-factors)
9. [Multifactor Investing](#9-multifactor-investing)
10. [You Are Already a Factor Investor](#10-you-are-already-a-factor-investor)
11. [Limitations of Factor Investing](#11-limitations-of-factor-investing)
12. [Factor Regression — Theory and Python](#12-factor-regression--theory-and-python)
13. [Factor Investing in India](#13-factor-investing-in-india)
14. [Applying PCA and Factor Analysis in Quant Finance](#14-applying-pca-and-factor-analysis-in-quant-finance)

---

# PART I: STATISTICAL FOUNDATIONS

---

## 1. Why Dimensionality Reduction? The Core Problem

### 1.1 The Setup

Given an $N \times D$ data matrix $X$ whose rows $\{x_i^\top\}_{i=1}^N$ are i.i.d. samples from some unknown distribution $p(x)$, we want to estimate that distribution. In finance, $N$ might be 252 trading days and $D$ might be 500 stocks or 50 financial variables.

The most natural probabilistic model is the **multivariate normal**:

$$p(x) = \mathcal{N}(x \mid \mu, \Sigma)$$

where $\mu \in \mathbb{R}^D$ is the mean vector and $\Sigma \in \mathbb{R}^{D \times D}$ is the covariance matrix.

**The problem:** a full Gaussian has

$$D + \frac{D(D+1)}{2} \approx O(D^2)$$

free parameters. For $D = 50$ assets this is already 1,325 parameters. For $D = 500$ stocks it's 125,250 parameters. Estimating this reliably requires enormous amounts of data — far more than financial time series typically provide.

### 1.2 The Solution: Latent Structure

High-dimensional financial data are almost never truly independent. Stock returns move together because of shared exposure to economic forces — interest rates, GDP growth, market sentiment. The key insight is:

> **Observed high-dimensional correlations are driven by a small number of hidden (latent) common causes.**

This is the premise behind every technique in this document. We don't need to estimate all $O(D^2)$ entries of $\Sigma$. We only need to find those few hidden causes — the *factors*.

---

## 2. Factor Analysis — The Probabilistic Foundation

### 2.1 The Generative Story

Factor Analysis (FA) is a **generative latent-variable model**. "Generative" means once parameters are learned, you can sample new synthetic observations. "Latent-variable" means each observed sample $x_i$ is explained by an unobserved variable $z_i$.

Imagine each stock's return is constructed as follows: a small number of hidden macroeconomic engines (the *factors*) fire with some magnitudes; those get mixed through a loading matrix and perturbed by idiosyncratic noise. That's the FA story.

**Intuitively:** If we have 500 stock returns, maybe only 3–5 underlying economic forces (market direction, interest rate sensitivity, growth vs. value sentiment) explain most of the comovement.

### 2.2 The Formal Model

For each observation $x_i \in \mathbb{R}^D$, the FA model says:

$$z_i \sim \mathcal{N}(0, I_L)$$

$$x_i \mid z_i, W, \mu, \Psi \sim \mathcal{N}(W z_i + \mu, \Psi)$$

which can be written in generative form as:

$$\boxed{x_i = W z_i + \mu + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \Psi)}$$

where $z_i$ and $\varepsilon_i$ are independent, and:

| Symbol | Dimension | Meaning |
|--------|-----------|---------|
| $z_i$ | $L \times 1$ | Latent factor vector (unobserved) |
| $W$ | $D \times L$ | **Factor loading matrix** |
| $\mu$ | $D \times 1$ | Mean vector |
| $\Psi$ | $D \times D$ | Diagonal noise covariance |
| $\psi_d$ | scalar | Noise variance for observed variable $d$ |

The key constraint is that $L \ll D$ (few factors, many observations) and $\Psi$ is **diagonal** — the noise in each observed variable is independent.

### 2.3 Why the Standard Normal Prior is No Loss of Generality

We always place $z_i \sim \mathcal{N}(0, I_L)$. Any other choice of mean or scale for the prior can be absorbed into $W$ and $\mu$. This is the *canonical* form of the model.

### 2.4 The Marginal Distribution — The Key Result

Since $z_i$ is unobserved, we integrate it out. Using the rules for linear Gaussian models:

$$\mathbb{E}[x_i] = W \mathbb{E}[z_i] + \mu = \mu$$

$$\text{Cov}(x_i) = W \underbrace{\text{Cov}(z_i)}_{= I_L} W^\top + \text{Cov}(\varepsilon_i) = WW^\top + \Psi$$

So the marginal distribution of $x_i$ is:

$$\boxed{p(x_i \mid W, \mu, \Psi) = \mathcal{N}(x_i \mid \mu, \underbrace{WW^\top + \Psi}_{=: C})}$$

**This is the payoff.** Instead of estimating a full $D \times D$ covariance matrix $C$ with $O(D^2)$ parameters, we model it as:

$$C = \underbrace{WW^\top}_{\text{low-rank, captures shared structure}} + \underbrace{\Psi}_{\text{diagonal, idiosyncratic noise}}$$

This has only $DL + D$ parameters — linear in both $D$ and $L$.

### 2.5 The Log-Likelihood

The model is fit by maximizing:

$$\mathcal{L} = \sum_{i=1}^N \log p(x_i \mid W, \mu, \Psi) = -\frac{N}{2}\log\det(2\pi C) - \frac{1}{2}\sum_{i=1}^N (x_i - \mu)^\top C^{-1}(x_i - \mu)$$

This is a function of $W$ and $\Psi$ only (since $\mu^* = \bar{x}$ is solved analytically).

### 2.6 Parameter Estimation

Optimizing the FA likelihood has no closed-form solution and requires **iterative algorithms**. Two main approaches exist:

**SVD-based approach** (used by scikit-learn):

1. Initialize $\Psi$ with small random positive values.
2. While log-likelihood has not converged:
   - Solve for $W$ given current $\Psi$ using SVD of the "whitened" covariance.
   - Update $\Psi$ to be the diagonal of the residual variance $C - WW^\top$.
   - Recompute log-likelihood.

**Expectation-Maximization (EM)**:

- **E-step:** Compute the posterior $p(z_i \mid x_i, W, \Psi)$ — the expected factor scores.
- **M-step:** Update $W$ and $\Psi$ to maximize the expected complete-data log-likelihood.

Both converge to a local maximum. The SVD approach is preferred for large $D$.

### 2.7 The Posterior — Inferring Factor Scores

Once $W, \mu, \Psi$ are estimated, we can infer the latent factor score for each observation:

$$p(z_i \mid x_i) = \mathcal{N}(z_i \mid m_i, S_i)$$

where:
$$S_i = (I + W^\top \Psi^{-1} W)^{-1}, \quad m_i = S_i W^\top \Psi^{-1}(x_i - \mu)$$

In finance: if $x_i$ is a vector of 500 stock returns on day $i$, then $m_i$ gives us the inferred values of the 3–5 underlying economic factors on that day.

### 2.8 Rotational Non-Identifiability

**A crucial subtlety:** FA solutions are not unique. For any orthogonal matrix $R$, replacing $W \to WR$ gives the same likelihood since:

$$(WR)(WR)^\top = WRR^\top W^\top = WW^\top$$

This **orthogonal rotational ambiguity** means:

- Different random initializations can give different-looking $W$.
- Comparing factor solutions across different runs is unreliable.
- Interpretation of factors is ambiguous without additional conventions.

This is why FA often applies **rotation schemes** (Varimax, Promax) to make factors interpretable — finding the rotation where each variable loads heavily on as few factors as possible.

### 2.9 Communality and Uniqueness

For variable $d$:

$$h_d^2 = \sum_{k=1}^L f_{dk}^2 = \text{communality}$$

$$u_d = 1 - h_d^2 = \psi_d / \text{Var}(x_d) = \text{uniqueness (factor uniqueness)}$$

Communality is the proportion of variance in variable $d$ explained by the common factors. Uniqueness (also called specificity) is the fraction that is pure idiosyncratic noise. In finance, a stock with high uniqueness has returns driven mostly by company-specific events rather than market-wide factors.

---

## 3. Probabilistic PCA — The Bridge to Classical PCA

### 3.1 PPCA as a Special Case of FA

**Probabilistic PCA (PPCA)** is obtained by restricting FA's noise covariance to be **isotropic**:

$$\Psi = \sigma^2 I_D$$

All observed dimensions share the same noise level. This single restriction enables a beautiful closed-form solution.

### 3.2 The Closed-Form Solution

Let $\mu^* = \frac{1}{N}\sum_i x_i$ be the sample mean and $S = \frac{1}{N}\sum_i(x_i-\mu^*)(x_i-\mu^*)^\top$ be the sample covariance with eigendecomposition $S = U\Lambda U^\top$.

The maximum likelihood estimates are:

$$\boxed{W^* = U_L (\Lambda_L - \sigma^2 I_L)^{1/2} R}$$

$$\boxed{\sigma^{2*} = \frac{1}{D-L}\sum_{j=L+1}^D \lambda_j}$$

where:

- $U_L$ contains the top $L$ eigenvectors of $S$
- $\Lambda_L = \text{diag}(\lambda_1, \ldots, \lambda_L)$ contains the top $L$ eigenvalues
- $R$ is any orthogonal matrix (rotational freedom)
- $\sigma^{2*}$ is the **average variance of the discarded dimensions**

**Interpretation:** The noise level equals the average unexplained variance per dimension. This is elegant: PPCA "uses up" the top $L$ directions to capture signal, and everything left over is attributed to uniform noise.

### 3.3 The Limit: PCA Recovered

As $\sigma^2 \to 0$, the PPCA solution recovers classical PCA:

- The loading matrix $W^*$ aligns with the top $L$ principal components.
- Reconstructed points collapse exactly onto the principal subspace.
- There is no shrinkage toward the mean.

When $\sigma^2 > 0$, there is regularization: reconstructed points are pulled slightly toward $\mu^*$ — a feature that's actually desirable in financial applications where returns are noisy.

### 3.4 The Hierarchy

$$\text{Factor Analysis} \supset \text{PPCA} \xrightarrow{\sigma^2 \to 0} \text{PCA}$$

| Method | Noise Structure | Solution | Key Use |
|--------|----------------|----------|---------|
| FA | Diagonal $\Psi$ (heteroscedastic) | Iterative | True factor models with variable-specific noise |
| PPCA | Isotropic $\sigma^2 I$ | Closed-form | Fast, clean probabilistic dimensionality reduction |
| PCA | None (deterministic) | Eigendecomposition | Variance maximization, visualization |

---

## 4. Principal Component Analysis (PCA)

### 4.1 The Goal

PCA seeks components $z = [z_1, z_2, \ldots, z_p]$ that are **linear combinations** of the original variables $x = [x_1, x_2, \ldots, x_p]$:

$$z = Ux$$

where $U$ is orthogonal, maximizing the variance of the components in sequence.

The first component $z_1 = u_1^\top x$ maximizes $\text{Var}(z_1)$ subject to $\|u_1\| = 1$. The second component maximizes variance while being uncorrelated with $z_1$, and so on.

### 4.2 The Solution via Eigendecomposition

The solution is found by the eigenvalue decomposition of the **correlation (or covariance) matrix**:

$$(R - \lambda I) u = 0$$

where $R$ is the sample correlation matrix, $\lambda$ is the eigenvalue, and $u$ is the eigenvector.

- **Eigenvalues** $\lambda_k$ are the variances of the associated components.
- **Eigenvectors** $u_k$ define the directions (loadings).
- The covariance matrix of the components is $D = \text{diag}(\lambda_1, \ldots, \lambda_p)$.

### 4.3 Factor Loadings

Factor loadings are the **correlations** between the original variables and the components:

$$F = \text{cov}(x, z) = D^{1/2} U$$

Concretely: $f_{dk} = \text{corr}(x_d, z_k) = u_{dk} \sqrt{\lambda_k}$.

The proportion of variance in variable $x_d$ explained by $c$ retained components is $\sum_{k=1}^c f_{dk}^2$. When all $p$ components are retained, this equals 1 (all variance explained).

### 4.4 Scale Sensitivity — Use Correlations, Not Covariances

PCA maximizes variance and is therefore **sensitive to scale differences** among variables. If one variable is measured in dollars and another in percentages, the dollar-denominated variable will dominate. Best practice: **standardize all variables** before running PCA (i.e., work with the correlation matrix $R$, not the covariance matrix $\Sigma$).

### 4.5 Component/Factor Retention Rules

Since PCA is a data reduction method, we need to decide how many components to keep. There is always a trade-off: parsimony (few components) vs. completeness (explaining most variance).

**Kaiser's Rule:** Retain only components with eigenvalue $\lambda_k > 1$. The rationale: a component should explain *at least as much variance as a single original variable* (which has variance 1 when standardized). This is the most widely used rule.

**Scree Plot:** Plot eigenvalues in descending order. Look for an "elbow" — a sharp bend beyond which eigenvalues level off. Components before the elbow are retained. This is more subjective but often identifies a better number than Kaiser's rule alone.

**Variance Explained:** Retain enough components to explain a desired proportion of total variance (e.g., 70%, 80%, 90%). The cumulative proportion is:

$$\text{Cumulative Proportion}_k = \frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^p \lambda_j}$$

### 4.6 Factor Rotation

The loading matrix is often **rotated** to achieve a "simple structure" — where each variable loads highly on one factor and near-zero on others. This makes factors interpretable as natural clusters of variables.

**Orthogonal Rotation** (rotated components remain uncorrelated):

- **Varimax:** Maximizes variance of squared loadings *within each factor* (across variables). Good for finding factors that each have a few highly loaded variables.
- **Quartimax:** Maximizes variance of squared loadings *within each variable* (across factors). Good for finding one dominant factor.

**Oblique Rotation** (allows correlated factors):

- **Promax:** Allows factor axes to be non-perpendicular, pursuing the goal of aligning factors as closely as possible to clusters of variables.
- Produces separate *pattern matrix* (regression coefficients) and *structure matrix* (correlations with factors).

**Important note:** Rotation does not change the overall variance explained; it only redistributes it among components to improve interpretability.

### 4.7 Component Scores

Once loadings are determined, we can compute a **score** for each observation on each component:

$$\hat{z}_i = B^\top x_i$$

where $B = R^{-1}F$ is the scoring coefficient matrix. These scores can substitute for the original variables in subsequent analysis — e.g., as regressors or as portfolio weights.

### 4.8 When Should You Use PCA?

PCA requires that the original variables share enough in common. The **Kaiser-Meyer-Olkin (KMO)** measure of sampling adequacy quantifies this:

$$\text{KMO} = \frac{\sum_{d \ne d'} r_{dd'}^2}{\sum_{d \ne d'} r_{dd'}^2 + \sum_{d \ne d'} p_{dd'}^2}$$

where $r_{dd'}$ are correlation coefficients and $p_{dd'}$ are partial correlation coefficients. Values above 0.5 are considered minimally acceptable; above 0.7 is good; above 0.8 is meritorious.

**Bartlett's sphericity test** checks whether the correlation matrix is significantly different from the identity (i.e., variables are not independent). The null hypothesis is that the correlation matrix equals $I$; rejection suggests PCA is appropriate.

---

## 5. PCA — Example and Interpretation

*This section uses a concrete example from Katchova (2013) on US Gross State Product data (50 states × 13 economic sectors).*

### 5.1 The Dataset

Variables: agriculture, mining, construction, durable manufacturing, non-durable manufacturing, transportation, communications, energy, wholesale trade, retail trade, real estate, services, and government — all expressed as shares of gross state product.

### 5.2 Eigenvalues and Variance Explained

| Component | Eigenvalue | Proportion of Variance | Cumulative |
|-----------|-----------|----------------------|------------|
| Comp1 | 3.24 | 0.25 | 0.25 |
| Comp2 | 2.24 | 0.17 | 0.42 |
| Comp3 | 1.96 | 0.15 | 0.57 |
| Comp4 | 1.36 | 0.10 | 0.68 |
| Comp5 | 1.16 | 0.09 | 0.77 |
| Comp6–13 | <1 | — | 1.00 |

**Reading this table:**

- There are always exactly $p = 13$ components (equal to the number of variables).
- The first 5 components have eigenvalues > 1 (Kaiser's rule → retain 5).
- The scree plot shows an elbow between components 3 and 5 → retain 3.
- The first 3 components explain 57% of total variation; adding components 4–5 brings it to 77%.

**Practical decision:** Given the scree plot elbow, 3 components are retained for parsimony. For completeness, 5 may also be justifiable.

### 5.3 Component Loadings

Selected loadings (correlations between components and original variables):

| Variable | Comp1 | Comp2 | Comp3 |
|----------|-------|-------|-------|
| Mining | **0.47** | 0.00 | -0.26 |
| Transp | **0.42** | 0.15 | 0.01 |
| TradeR | -0.09 | 0.26 | **0.51** |
| RE | **-0.36** | 0.03 | **-0.45** |
| Services | **-0.38** | **0.38** | -0.13 |
| Govt | 0.29 | **0.37** | 0.09 |
| Constr | 0.04 | **0.39** | 0.26 |

**Convention:** Loadings above |0.3| are considered meaningful.

**Naming the components:**

- **Component 1:** High positive loadings on Mining and Transport; high negative loadings on Services and Real Estate → *"Resource-extraction economy"* vs. *"Services-based economy"*.
- **Component 2:** High positive loadings on Construction, Services, Government → *"Public-sector + construction"* axis.
- **Component 3:** High positive loading on Retail Trade; high negative loading on Real Estate → *"Consumer retail"* vs. *"Real estate"* axis.

This naming is interpretive — it's the analyst's job to give meaning to components based on which variables load heavily.

### 5.4 Component Score Plot

Plotting the scores of each US state on Components 1 and 2 reveals:

- **Alaska (AK)** and **Wyoming (WY)** are clear outliers with high scores on Component 1 — reflecting their dominance in mining and resource extraction.
- Most states cluster near the origin with relatively similar economic structures.
- This is a key diagnostic: outlier observations may unduly influence factor solutions.

### 5.5 Biplot Interpretation

A biplot simultaneously shows:

- **Observations** (states) as points.
- **Variables** as arrows (direction and length encode the loading).

Arrows pointing in the same direction: variables are positively correlated. Arrows pointing in opposite directions: negative correlation. Arrow length: contribution to the displayed components. States near an arrow tip score high on that variable.

### 5.6 Factor Analysis vs. PCA — What Changes?

When Factor Analysis is applied to the same data:

| | PCA | Factor Analysis |
|---|---|---|
| **Goal** | Explain variance | Explain covariance via latent factors |
| **Assumption** | None | Common factor model |
| **Components** | Principal components | Common factors + specific factors |
| **Uniqueness** | Not defined | $u_d = 1 - h_d^2$ |
| **Rotation** | Optional | Standard practice |

The **factor uniqueness** column in FA shows how much variance remains unexplained. For example, high uniqueness on Agriculture and Communications suggests those sectors have large idiosyncratic variation not captured by the 3–5 common factors.

### 5.7 Rotation Effects

The **Varimax** and **Promax** rotations redistribute loadings to achieve simple structure:

- After rotation, each variable tends to load on one or two factors rather than spreading across all.
- The Varimax (orthogonal) and Promax (oblique) rotations give similar results in this example — suggesting factors are only mildly correlated even when oblique rotation is allowed.
- The overall variance explained does not change; only the distribution across factors changes.

---

## 6. Exploratory Factor Analysis and Its Limitations

### 6.1 The Common Factor Model

The classical EFA model decomposes each observed variable $x_d$ as:

$$x_d = \lambda_{d1}\xi_1 + \lambda_{d2}\xi_2 + \cdots + \lambda_{dc}\xi_c + \delta_d$$

where:

- $\xi_k$ are **common factors** — shared across all variables.
- $\lambda_{dk}$ are **factor loadings** — the strength of factor $k$'s effect on variable $d$.
- $\delta_d$ is the **specific (unique) factor** — idiosyncratic to variable $d$.

**Assumptions:**

1. Common factors are mutually uncorrelated: $\text{Cov}(\xi_k, \xi_{k'}) = 0$ for $k \ne k'$.
2. Specific factors are mutually uncorrelated: $\text{Cov}(\delta_d, \delta_{d'}) = 0$.
3. Common and specific factors are uncorrelated: $\text{Cov}(\xi_k, \delta_d) = 0$.

### 6.2 EFA vs. PCA — The Conceptual Difference

This is one of the most commonly confused distinctions in statistics:

| | EFA | PCA |
|---|---|---|
| **Model** | Common factor model (latent causes) | Linear combination (summary statistic) |
| **Direction** | Causes → Observations | Observations → Components |
| **Unexplained variance** | Explicit (uniqueness $u_d$) | None (all variance accounted for) |
| **Purpose** | Discover latent constructs | Reduce dimensionality |
| **Best for** | Psychology, economics (latent traits) | Signal compression, feature extraction |

**Key intuition:** EFA asks *"what latent things could have generated this data?"* PCA asks *"what linear combination of my data captures the most variance?"* Both are useful; they answer different questions.

### 6.3 Limitations of Exploratory Factor Analysis

**1. The rotation problem.** Factor solutions are only identified up to orthogonal rotation. There is no "true" loading matrix — only conventions. Different rotation methods can give strikingly different-looking factors with equally good fit to the data.

**2. Number of factors is ambiguous.** Kaiser's rule, scree plots, and parallel analysis often disagree. Retaining different numbers of factors changes the interpretation entirely.

**3. Data requirements.** EFA requires sufficient sample size and sufficient intercorrelation among variables (KMO > 0.5). Financial returns often have changing correlation structure (regimes), violating the stationarity assumption.

**4. Assumed linear relationships.** FA and PCA both assume that the relationship between variables and factors is linear. Non-linear dependencies (common in options, credit, or crisis periods) are missed.

**5. Stationarity assumption.** FA assumes the factor structure is stable over time. In finance, factor loadings are demonstrably time-varying — a stock's sensitivity to "growth" may change with its business cycle. This motivates **rolling PCA** and **dynamic factor models**.

**6. Overfitting and data mining.** With enough factors, any covariance matrix can be fit perfectly. Cross-validation or out-of-sample testing is essential to validate factor solutions.

**7. Interpretability is subjective.** Naming factors is an art, not a science. Two analysts can give entirely different names to the same component and both be defensible. This is especially problematic in finance where factor names carry regulatory and commercial implications.

---

# PART II: FACTORS IN FINANCE

---

## 7. Factors in Investing — History and Theory

### 7.1 What Is a Factor?

A **factor** (also called a *risk premium*, *risk factor*, or *style factor*) is:

> A quantitative characteristic shared across a set of securities that constitutes an underlying, independent source of return and risk.

The word "independent" is key: a true factor should carry return that cannot be explained by exposure to other known factors. The word "risk" is equally important: factors are compensated because they carry systematic risk — bearing them is unpleasant in some state of the world.

**The food analogy:** A pizza and a burger look different but are nutritionally similar — both calorically dense, high in fat. Similarly, two funds might look different (different names, different providers, different styles) but might be loading on the same underlying factor. A factor regression sees through the packaging to the underlying exposures, just like a nutrition label.

**Factors exist in all asset classes:**

- **Equities:** Market beta (ERP), Size, Value, Momentum, Profitability, Investment, Low Volatility.
- **Bonds:** Term (duration), Credit.
- **Currencies:** Carry, Momentum.
- **Commodities:** Carry, Momentum, Backwardation/Contango.

This document focuses on equity factors.

### 7.2 The Origin: CAPM (1964–1966)

The **Capital Asset Pricing Model** was the first single-factor asset pricing model, developed by William Sharpe (1964), John Lintner (1965), and Jan Mossin (1966) — building on Harry Markowitz's mean-variance framework.

**The CAPM equation:**

$$\mathbb{E}[R_i] = R_f + \beta_i \left(\mathbb{E}[R_m] - R_f\right)$$

where:

- $R_i$: Return on asset $i$
- $R_f$: Risk-free rate (e.g., 30-day T-bill)
- $R_m$: Return on the market portfolio
- $\beta_i = \frac{\text{Cov}(R_i, R_m)}{\text{Var}(R_m)}$: Market beta

**Intuition:** The only source of systematic risk is exposure to the market portfolio. All other variation is idiosyncratic and can be diversified away. An asset's expected return is completely determined by its beta.

**The Equity Risk Premium (ERP) in CAPM:**

Why does the market portfolio pay a premium over T-bills? Because it is risky — it crashes in recessions precisely when marginal utility of wealth is highest. Investors demand compensation for holding this undesirable risk. The ERP = $\mathbb{E}[R_m] - R_f > 0$ is this compensation.

**CAPM's empirical failures:**

- Low-beta stocks earn higher risk-adjusted returns than predicted (Low Volatility anomaly).
- Small stocks earn more than large stocks (not explained by beta alone).
- Cheap stocks earn more than expensive stocks (value anomaly).
- Past winners outperform (momentum).

These failures motivated multifactor models.

### 7.3 Arbitrage Pricing Theory (APT, 1976)

Stephen Ross's **APT** (1976) was the first formal *multifactor* asset pricing theory. It requires only one assumption: no arbitrage.

**The APT equation:**

$$R_i = \alpha_i + \beta_{i1}F_1 + \beta_{i2}F_2 + \cdots + \beta_{iK}F_K + \varepsilon_i$$

where $F_k$ are systematic factors and $\varepsilon_i$ is idiosyncratic noise. In a no-arbitrage equilibrium:

$$\mathbb{E}[R_i] = R_f + \sum_{k=1}^K \beta_{ik} \lambda_k$$

where $\lambda_k$ is the **factor risk premium** for factor $k$.

**APT's strengths:**

- More flexible than CAPM — multiple factors allowed.
- Derived from no-arbitrage alone (weaker assumption than CAPM's market clearing).

**APT's weakness:**

- APT does not specify *which* factors matter. It is silent on the identity, number, or interpretation of the factors. This is where empirical research fills the gap.

### 7.4 Fama-French Three-Factor Model (1992–1993)

Eugene Fama and Ken French's seminal paper *"The Cross-Section of Expected Stock Returns"* (Journal of Finance, 1992) documented that **two additional factors** — beyond market beta — explain stock returns:

$$R_i - R_f = \alpha + \beta_{\text{MKT}}(R_m - R_f) + \beta_{\text{SMB}} \cdot \text{SMB} + \beta_{\text{HML}} \cdot \text{HML} + \varepsilon_i$$

**The three factors:**

| Factor | Name | Long Side | Short Side | Intuition |
|--------|------|-----------|------------|-----------|
| $R_m - R_f$ | Market (ERP/Beta) | Equities | T-bills | Risk of holding equity |
| SMB | Small Minus Big | Small-cap stocks | Large-cap stocks | Small firms are riskier |
| HML | High Minus Low | High book/price | Low book/price | "Value" firms are riskier |

**Construction of SMB and HML (long-short portfolios):**

- All stocks are sorted by size (market cap) into Small and Big halves.
- All stocks are sorted by book-to-market (B/M) into Value (top 30%), Neutral (40%), Growth (bottom 30%).
- SMB = average return of small-cap portfolios minus average return of large-cap portfolios.
- HML = average return of high B/M portfolios minus average return of low B/M portfolios.

**Empirical performance:** The three-factor model explained approximately 90% of the cross-sectional variation in average returns (vs. ~70% for CAPM). The value premium (HML) averaged about 4–5% per year historically.

**Why might value stocks earn higher returns?** Two theories:

1. **Risk-based (Fama-French):** Value firms are financially distressed, have more assets-in-place (less flexible), and do worse in recessions → rational risk premium.
2. **Behavioral (Lakonishok, Shleifer, Vishny 1994):** Investors extrapolate past earnings too far, making glamour (growth) stocks overpriced → mispricing premium.

### 7.5 The Carhart Four-Factor Model (1997)

Mark Carhart added **momentum** to the Fama-French three-factor model, motivated by Jegadeesh and Titman's (1993) finding that past winners continue to outperform for 3–12 months:

$$R_i - R_f = \alpha + \beta_{\text{MKT}}(R_m-R_f) + \beta_{\text{SMB}}\cdot\text{SMB} + \beta_{\text{HML}}\cdot\text{HML} + \beta_{\text{UMD}}\cdot\text{UMD} + \varepsilon_i$$

| Factor | Name | Meaning |
|--------|------|---------|
| UMD | Up Minus Down | Past-12-month winners minus losers (skip most recent month) |

**Momentum's outperformance:** Historically ~7–8% per year gross of transaction costs. However, momentum is a high-turnover, high-transaction-cost strategy and suffers from catastrophic "momentum crashes" during sharp market reversals (e.g., 2008–2009, March 2020).

**Is momentum a risk factor or a mispricing?** This is debated. There is no clear risk story for why recent winners should be *riskier*. Most researchers lean toward behavioral explanations: anchoring, conservatism, and underreaction to news cause prices to gradually incorporate information, creating momentum.

### 7.6 Fama-French Five-Factor Model (2015)

Fama and French (2015) extended the three-factor model by adding **profitability** and **investment** factors:

$$R_i - R_f = \alpha + \beta_1(R_m-R_f) + \beta_2\text{SMB} + \beta_3\text{HML} + \beta_4\text{RMW} + \beta_5\text{CMA} + \varepsilon$$

| Factor | Name | Long | Short | Empirical Premium |
|--------|------|------|-------|------------------|
| RMW | Robust Minus Weak | High gross profit / equity | Low | ~3–4% p.a. |
| CMA | Conservative Minus Aggressive | Low asset growth | High | ~3% p.a. |

**Profitability (RMW):** Novy-Marx (2013) showed that gross profitability (gross profit / total assets) is a powerful predictor of returns, even after controlling for value. The intuition from the investment CAPM: high-profitability firms have high discount rates (risky) and have rationally chosen to invest less than their lower-profitability peers.

**Investment (CMA):** Firms that invest aggressively (grow their asset base rapidly) tend to underperform. From the investment CAPM perspective: firms invest when their cost of capital is *low* — high investment signals low discount rate → low expected return for investors. The rational story: low-investment firms are riskier because they have fewer growth options to buffer them in bad times.

**Note:** In the five-factor model, HML becomes largely redundant once RMW and CMA are included — suggesting that value captures something already explained by profitability and investment.

### 7.7 The AQR / Quality Factor

AQR Capital Management (Asness, Frazzini, Pedersen) extended factor research with two additional factors:

**QMJ — Quality Minus Junk (2019):**
Quality is a composite of profitability (ROE, gross profitability), growth (5-year growth in profitability measures), and safety (low beta, low leverage, low bankruptcy risk). The QMJ factor goes long high-quality firms and short "junk" firms.

**BAB — Betting Against Beta (2014):**
Perhaps the most counterintuitive factor: low-beta stocks earn higher *risk-adjusted* returns than high-beta stocks — violating CAPM. The explanation: leverage-constrained investors (who can't use leverage) tilt toward high-beta stocks to boost returns, inflating their prices. Low-beta stocks are underdemanded → underpriced → higher returns. The BAB factor explicitly exploits this by going long low-beta stocks (levered) and short high-beta stocks.

### 7.8 Summary of Major Equity Factors

| Factor | Symbol | Definition | Annualized Premium (US, historical) | Risk or Mispricing? |
|--------|--------|-----------|--------------------------------------|---------------------|
| Market | MKT | $R_m - R_f$ | ~5–7% | Risk |
| Size | SMB | Small - Large | ~2–3% (after junk control) | Risk |
| Value | HML | High B/M - Low B/M | ~4–5% | Both debated |
| Momentum | UMD/WML | Winners - Losers | ~7–8% gross | Behavioral |
| Profitability | RMW / QMJ | Profitable - Unprofitable | ~3–4% | Risk (Investment CAPM) |
| Investment | CMA | Conservative - Aggressive | ~3% | Risk (Investment CAPM) |
| Low Volatility | BAB/LVOL | Low beta - High beta | ~3–4% | Behavioral/Constraints |

*All premia are gross of transaction costs, long-short, historical in US data. Out-of-sample premia will be lower.*

---

## 8. The Economic Mechanisms Behind Factors

Understanding *why* factors earn premia is crucial — without an economic mechanism, a factor is just data mining.

### 8.1 Risk-Based Explanations: The Consumption CAPM

The fundamental insight: investors distinguish **good times** (high income, high consumption) from **bad times** (recessions, job losses). Because marginal utility is concave, an extra dollar is worth much less in good times than in bad times.

Any asset that *pays off poorly in bad times* must offer a **higher expected return** to compensate investors for holding it.

Empirically (Belo, 2022), the beta of factor portfolios to consumption growth:

- **Book-to-Market (Value) spread:** Much higher consumption beta than the average stock.
- **Size spread:** Higher consumption beta than average.
- **Momentum spread:** Higher consumption beta.
- **Investment spread:** Higher consumption beta.

All four factor portfolios have higher compensation because they are *more exposed to consumption risk* than the average firm.

### 8.2 Risk-Based Explanations: The Investment CAPM

The investment CAPM (Zhang, 2005; Hou, Xue, Zhang, 2015) rests on the firm's optimization problem:

$$\text{Return} \propto \frac{\text{Profitability}}{\text{Cost of Investment}}$$

Three insights:

1. **High-profitability firms** invested less (stopped investing earlier because high cost of capital), signaling they are riskier → higher expected return.
2. **Low-investment firms** ran out of profitable opportunities before their peers, revealing a high cost of capital → higher expected return.
3. **Value firms** have more assets-in-place (locked investments). In recessions, they are forced to sell fixed assets at a loss, creating severe downside → higher risk premium.

### 8.3 Behavioral Explanations

The behavioral finance view is that *irrational, predictable human biases create mispricing*:

**Anchoring:** Investors over-weight the first information they receive about a stock (initial price, initial story). New contradicting information is processed slowly.

**Conservatism:** When presented with new information, investors revise their beliefs too slowly — updating at roughly half the rational rate. This creates under-reaction to earnings announcements → momentum.

**Representativeness heuristic:** Investors oversimplify by bundling similar things together. The most dangerous manifestation: **mistaking a good company for a good investment**. A great company (low risk) should have a *lower* expected return (less compensation needed). Investors who chase "great companies" bid up their prices and earn lower returns → value premium.

**Limits to arbitrage:** Even if mispricings are known, they persist because:

- Transaction costs of shorting overpriced stocks are high.
- Institutional constraints (benchmarking, career risk, redemption risk) prevent arbitrageurs from taking large positions.
- Short selling requires locating shares to borrow.

### 8.4 Checklist for a Real Factor

Both the risk-based and behavioral arguments agree on what makes a *real* factor vs. data mining:

| Criterion | Description |
|-----------|-------------|
| **Plausible economic mechanism** | There must be a logical reason why this factor earns a premium — not just a backtest |
| **Pervasiveness** | Factor should work across geographies, time periods, and asset classes |
| **Persistence** | Factor premium should survive after publication (post-1993 momentum, post-1992 value) |
| **Robustness** | Premium should hold with different measurement approaches |
| **External replication** | Multiple independent research groups should find the same result |
| **Net of costs** | Premium should survive realistic transaction costs |

**Red flag:** A factor based on a single backtest over a short period, using a new data source, with no economic story — is almost certainly data mining.

---

## 9. Multifactor Investing

### 9.1 Why Combine Factors?

Individual factors can underperform for extended periods:

- Value underperformed for nearly a decade (2007–2017 in the US).
- Size premium essentially disappeared after controlling for quality.
- Momentum crashes violently in sharp market reversals.

But factors tend to underperform *at different times* — their correlation is less than 1. A portfolio diversified across factors therefore has a more stable, reliable return profile.

**Diversification benefit across factors (approximate historical correlations):**

| | MKT | SMB | HML | MOM | RMW | CMA |
|---|---|---|---|---|---|---|
| MKT | 1 | 0.3 | -0.3 | -0.1 | -0.2 | -0.4 |
| SMB | | 1 | 0.1 | 0.0 | -0.4 | 0.0 |
| HML | | | 1 | -0.6 | 0.2 | 0.7 |
| MOM | | | | 1 | 0.1 | -0.3 |
| RMW | | | | | 1 | 0.1 |

Note the strongly negative correlation between HML (value) and MOM (momentum) — value and momentum diversify each other particularly well.

### 9.2 Factor Timing — The Graveyard of Good Ideas

One might think: "If I know value will underperform for a decade, shouldn't I time my factor exposure?" The overwhelming evidence says: **factor timing is extremely difficult and rarely adds value.**

- Factor valuation spreads (e.g., how cheap value stocks are relative to growth) have some predictive power but the timing signal is noisy and slow.
- Transaction costs from frequent switching often erode any timing benefit.
- Behavioral biases (recency bias) cause investors to exit factors at their cheapest (worst expected return time) — exactly wrong.

**The robust recommendation:** Maintain diversified multi-factor exposure through full cycles. Don't time.

### 9.3 Long-Short vs. Long-Only Factor Implementation

**Long-Short (as in academic factor portfolios):**

- Buy long the "good" decile and short the "bad" decile.
- Return is the *pure* factor premium — fully neutralizes market exposure.
- Requires shorting ability and leverage.
- As in, SMB = (average return of small stocks) − (average return of large stocks).

**Long-Only (as in factor ETFs):**

- Simply hold stocks that rank highly on the desired factors (overweight relative to the market).
- Return is dominated by market beta (the factor loading is smaller).
- No shorting required; accessible to all investors.
- The factor premium is delivered *in addition to* market exposure.

A long-only factor ETF will show both a high market beta (~1) and a positive factor loading (e.g., ~0.3–0.5 on value or size) in a factor regression — because it's holding the market plus an overlay.

### 9.4 Jointly Targeting Factors

Value and profitability have a natural synergy: value is short momentum (cheap stocks often have negative momentum) and profitability screens can mitigate this by filtering out *cheap and unprofitable* stocks ("value traps"). Mechanically:

- A pure value screen buys stocks that are cheap regardless of quality.
- Adding a profitability screen removes the cheapest *junk* — stocks that are cheap because they deserve to be cheap.
- The resulting portfolio has lower correlation to pure value but better risk-adjusted performance.

AQR's research and the Avantis/DFA approach explicitly target value and profitability jointly for this reason.

### 9.5 The "Next Generation" Factors Problem

A cautionary tale: the proliferation of claimed "new factors" in the past decade:

**Innovation Factor (R&D / Revenue):** One major provider claimed a ~6% annual alpha from tilting to high-R&D firms. However, academic literature studying the same characteristic over long-term US data finds **no statistically significant effect** (monthly return: −4 to −8 basis points, insignificant). The provider's backtest used only a single decade — far too short to establish a factor.

**Sustainability/ESG Factors:** With access to 25 ESG metrics from three providers, it is trivially easy to cherry-pick the one that happened to perform well in a given sample. After adjusting for exposure to the existing six Workhorse factors and looking at all 25 metrics jointly, the average value added is **−1.7% per year** — negative. The star ESG characteristic looks good only in isolation; it captures nothing beyond existing factor exposure.

**The diagnostic test:** Does the factor add value *after controlling for the six Workhorse factors (value, size, profitability, investment, momentum, low volatility)?* Does it work across *all* related characteristics (not just the one selected)? If not, it's data mining.

---

## 10. You Are Already a Factor Investor

### 10.1 The Uncomfortable Truth

Whether you realize it or not, if you hold a diversified portfolio, your returns are dominated by a small number of known risk factors. This is true for:

- Actively managed mutual funds
- Private portfolio managers
- Your own stock picks (if you hold 30+ stocks)
- Dividend funds
- Equal-weight index funds

**Running a factor regression on your portfolio will show you the truth.** Almost every actively managed fund's returns can be substantially explained by its loadings on the Fama-French 5 factors + momentum.

### 10.2 The Arithmetic of Active Management (Sharpe, 1991)

William Sharpe's famous paper established an iron law:

**Proposition 1:** Before costs, the return on the average actively managed dollar = the return on the average passively managed dollar.

*Proof:* The market is every investment combined. The "average" of all active portfolios plus the passive portfolio = the market. The passive portfolio earns the market return. Therefore active on average earns the market return. This is simply arithmetic — how averages work.

**Proposition 2:** After costs, the return on the average actively managed dollar < the return on the average passively managed dollar.

*Proof:* Active management has higher costs (management fees, transaction costs, bid-ask spreads). Since the average active portfolio earns the same gross return as the passive portfolio but incurs higher costs, the net return must be lower on average.

**The implications are profound:**

- The average active manager underperforms after fees.
- For an active manager to outperform, another active manager must underperform by the same amount.
- There is no reliable way for retail investors to identify the future winners in advance.

### 10.3 Equal-Weighted vs. Cap-Weighted Index Funds

A popular argument: "Equal weighting gives you more diversification and better returns than cap-weighting." Let's fact-check this with factor regression.

**Observation:** Yes, equal-weighted ETFs (e.g., RSP) often *do* show higher returns than cap-weighted ETFs (e.g., SPY) in historical backtests.

**Factor regression reveals:** The higher return is entirely explained by:

- Higher market beta (slightly more volatile → more exposure to MKT).
- Higher loading on SMB (equal-weighting tilts toward smaller stocks).
- Higher loading on HML (equal-weighting tilts toward cheaper stocks — large-cap growth stocks get underweighted).

**After controlling for these known factor exposures, the alpha is zero or negative.** You're not getting superior stock-picking. You're getting a free-rider on the size and value premiums — in a more expensive and less efficient wrapper.

**The lesson:** Before attributing outperformance to "strategy skill," always check whether factor exposures explain it.

### 10.4 Dividend Funds

Another emotional debate: "Dividend stocks give me cash flow; they must be better investments."

**Behavioral mistake:** In a frictionless pre-tax world, dividend policy is irrelevant (Modigliani-Miller). A firm that pays a $1 dividend leaves the investor $1 poorer in share price — you're just liquidating part of your position. On a post-tax basis, dividends are *worse* than capital gains because they trigger immediate ordinary income tax.

**Factor regression reveals:** Dividend fund outperformance is explained by their loading on:

- **Value (HML):** Dividend-payers tend to be cheap stocks (high earnings relative to price).
- **Profitability (RMW):** Dividend-payers tend to be profitable firms.

**The alpha is zero.** If you want the value and profitability premiums, harvest them directly and efficiently — not through a dividend wrapper that comes with unnecessary tax drag.

### 10.5 Warren Buffett's Alpha

Buffett's Alpha (Frazzini, Kabiller, Pedersen, 2018) showed that Berkshire Hathaway's legendary returns can be substantially explained by:

- High loading on market beta
- High loading on BAB (low-volatility/quality)
- High loading on QMJ (quality)
- Leverage (approximately 1.7× on average)

After controlling for these factors plus leverage, Buffett's alpha was **slightly positive but statistically indistinguishable from zero**. The conclusion is not that Buffett has no skill — but that his investment style (buying high-quality, cheap companies with leverage) is itself a systematic factor strategy that could in principle be replicated (though not perfectly, for many reasons).

---

## 11. Limitations of Factor Investing

### 11.1 The Zoo of Factors

Harvey, Liu, and Zhu (2016) catalogued over 300 claimed "factors" in the literature. Most are likely false positives from data mining. The threshold for statistical significance should be much higher (t-statistic > 3.0, not the conventional 2.0) to account for multiple testing.

### 11.2 Factor Capacity Constraints

Factors have capacity limits. As more money chases a factor premium, competition increases, bid-ask spreads widen, and returns compress. This is especially severe for:

- **Small cap** (liquidity is limited; large funds can't hold 50% of a small company's float).
- **Momentum** (requires frequent rebalancing; impact costs are large at scale).
- **Value** (slower turnover, but "value traps" and information effects matter).

### 11.3 The Skill vs. Factor Exposure Problem

Modern multifactor models explain ~95% of the differences in returns between two diversified portfolios. For most active managers, what remains is noise, not skill. The bar for declaring a manager has genuine alpha (beyond factor exposure) is extremely high.

### 11.4 Factor Drawdowns Can Be Devastating

| Factor | Worst Drawdown | Duration |
|--------|---------------|---------|
| Value (HML) | ~40% | 2007–2020 |
| Momentum (UMD) | ~60% (2009 crash) | 1 month |
| Size (SMB) | ~30% | 1980s–1990s |

These are long-short, gross-of-cost drawdowns. Real investor experience with long-only factor tilts was less extreme, but still severe. Behavioral abandonment at the worst moment is the primary way investors fail to capture factor premia.

### 11.5 Transaction Costs and Factor Decay

The gross factor premium must be large enough to survive:

- **Management fees** (0.05–0.50% for factor ETFs; 1–2% for actively managed alternatives).
- **Bid-ask spreads** (significant for small-cap value, momentum strategies).
- **Market impact costs** for large funds.
- **Short borrowing costs** for long-short implementations.

Net-of-cost premia are meaningfully smaller than the academic long-short figures.

### 11.6 Factor Exposure is Not Constant

Factor loadings in a factor regression are computed over a historical window and assumed stable. In reality:

- A value stock can become a growth stock; its factor loading changes.
- Macroeconomic regimes (rising rates, tech boom) shift which factors are in favor.
- A fund manager can drift their style over time (style drift).

This motivates **rolling factor regressions** — see Section 12.

### 11.7 Emerging Market and India-Specific Limitations

- Factor premia documented in US data do not always replicate in emerging markets with the same magnitude.
- Liquidity is lower; transaction costs higher.
- Short selling restrictions in markets like India limit long-short implementations.
- Corporate governance and regulatory differences affect profitability and investment factors.
- Factor data (clean accounting data) quality is lower in many emerging markets.

---

# PART III: PRACTICAL APPLICATIONS

---

## 12. Factor Regression — Theory and Python

### 12.1 The Factor Model Equation

The general multifactor regression is:

$$R_{i,t} - R_{f,t} = \alpha_i + \sum_{k=1}^K \beta_{ik} F_{k,t} + \varepsilon_{i,t}$$

where:

- $R_{i,t} - R_{f,t}$: **Excess return** of asset/fund $i$ at time $t$ (always use excess returns, not raw returns).
- $\alpha_i$: **Alpha** — the intercept; return unexplained by factor exposure.
- $\beta_{ik}$: **Factor loading** (beta) on factor $k$ — sensitivity to that factor.
- $F_{k,t}$: Return of factor $k$ portfolio at time $t$.
- $\varepsilon_{i,t}$: Idiosyncratic residual.

**Alpha interpretation:**

- $\alpha > 0$: Manager/fund outperforms after adjusting for factor exposure → genuine skill (or unexplained anomaly).
- $\alpha < 0$: Manager/fund underperforms on risk-adjusted basis → destroys value.
- $\alpha \approx 0$: Returns fully explained by factor exposure → no skill, just factor loading.

**Factor loading interpretation:**

- $\beta_{\text{SMB}} = 0.4$: Fund earns 40% of the size premium; tilted toward small caps.
- $\beta_{\text{HML}} = -0.5$: Fund is anti-value (tilted toward growth stocks).
- $\beta_{\text{UMD}} = 0.6$: Fund benefits from momentum; implicitly buying recent winners.

### 12.2 Python Implementation — Full Code

```python
# ============================================================
# FACTOR REGRESSION IN PYTHON
# Using Fama-French Five-Factor + Momentum (Six-Factor Model)
# ============================================================

# --- 1. IMPORTS ---
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import get_famafrench_factors as gff  # pip install get-famafrench-factors

# ============================================================
# --- 2. SOURCE FUND DATA ---
# ============================================================
def get_fund_returns(tickers: list, start="2000-01-01") -> pd.DataFrame:
    """Download monthly returns for a list of ETF/stock tickers."""
    rets = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    rets = rets.to_period("M")         # Convert to monthly period
    rets = rets.pct_change().dropna()  # Percent change → returns
    return rets

# ============================================================
# --- 3. SOURCE FAMA-FRENCH FACTOR DATA ---
# ============================================================
def get_six_factor_model() -> pd.DataFrame:
    """Build a six-factor model: FF5 + Momentum."""
    ff5 = gff.famaFrench5Factor().set_index("date").to_period("M")
    mom = gff.momentumFactor().set_index("date").to_period("M")
    
    # Merge and reorder so RF is last
    model = ff5.join(mom)
    cols_no_rf = [c for c in model.columns if c != "RF"]
    model = model[cols_no_rf + ["RF"]]
    
    return model

# ============================================================
# --- 4. FACTOR REGRESSION ---
# ============================================================
def factor_regression(rets: pd.DataFrame, factor_model: pd.DataFrame) -> pd.DataFrame:
    """
    Run OLS factor regressions for each security in rets.
    
    Returns a DataFrame of factor loadings (betas) and alpha.
    Columns = tickers, Rows = factors + alpha + annualized_alpha.
    """
    # --- Align indices ---
    common_idx = rets.index.intersection(factor_model.index)
    rets = rets.loc[common_idx]
    factor_model = factor_model.loc[common_idx]
    
    # --- Compute excess returns ---
    rf = factor_model["RF"].values
    rets_excess = rets.subtract(rf, axis=0)
    
    # --- Factor model without RF ---
    factors = factor_model.drop(columns=["RF"]).copy()
    
    # --- Add intercept column for alpha ---
    factors["alpha"] = 1.0
    
    # --- Run OLS ---
    lm = sm.OLS(rets_excess, factors).fit()
    
    # --- Extract factor loadings ---
    loadings = lm.params.copy()
    
    # --- Compute annualized alpha ---
    alpha_per_period = loadings.loc["alpha"]
    annualized_alpha = ((1 + alpha_per_period) ** 12 - 1) * 100  # percentage
    
    # --- Format output ---
    loadings = loadings.round(3)
    loadings = loadings.astype(object)
    loadings.loc["alpha"] = (alpha_per_period * 100).round(3).astype(str) + "%"
    
    ann_alpha_row = pd.Series(
        (annualized_alpha.round(2).astype(str) + "%"),
        name="annualized_alpha"
    )
    
    result = pd.concat([loadings, ann_alpha_row.to_frame().T])
    result.columns = rets.columns
    
    return result


# ============================================================
# --- 5. ROLLING FACTOR REGRESSION ---
# ============================================================
def rolling_factor_regression(
    rets: pd.DataFrame,
    factor_model: pd.DataFrame,
    roll_window: int = 36
) -> dict:
    """
    Rolling OLS factor regression with a sliding window.
    
    Returns a dict {ticker: DataFrame of loadings over time}.
    """
    # Align data
    common_idx = rets.index.intersection(factor_model.index)
    rets = rets.loc[common_idx]
    factor_model = factor_model.loc[common_idx]
    
    n_periods = len(rets)
    
    if n_periods <= roll_window:
        print(f"Sample ({n_periods} periods) smaller than roll_window ({roll_window})")
        return {}
    
    # Compute excess returns
    rf = factor_model["RF"].values
    rets_excess = rets.subtract(rf, axis=0)
    factors = factor_model.drop(columns=["RF"]).copy()
    factors["alpha"] = 1.0
    
    # Generate rolling windows
    windows = [(s, s + roll_window) for s in range(n_periods - roll_window + 1)]
    
    results = {}
    for ticker in rets.columns:
        loadings_per_window = []
        
        for win_start, win_end in windows:
            y = rets_excess.iloc[win_start:win_end][ticker]
            X = factors.iloc[win_start:win_end]
            lm = sm.OLS(y, X).fit()
            
            entry = lm.params.to_dict()
            entry["date"] = rets.index[win_end - 1]
            loadings_per_window.append(entry)
        
        df = pd.DataFrame(loadings_per_window).set_index("date")
        results[ticker] = df
    
    return results


# ============================================================
# --- 6. MAIN USAGE EXAMPLE ---
# ============================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Define tickers
    tickers = ["VTI", "AVUV", "VLU", "MTUM"]
    
    # Fetch data and factor model
    rets = get_fund_returns(tickers, start="2013-01-01")
    model = get_six_factor_model()
    
    # --- Static regression ---
    print("=== Static Factor Regression ===")
    result = factor_regression(rets, model)
    print(result)
    
    # --- Rolling regression ---
    print("\n=== Rolling Factor Regression (36-month window) ===")
    rolling = rolling_factor_regression(rets, model, roll_window=36)
    
    # Plot rolling loadings for each ticker
    for ticker, df in rolling.items():
        fig, ax = plt.subplots(figsize=(12, 5))
        df.drop(columns=["alpha"], errors="ignore").plot(ax=ax)
        ax.set_title(f"Rolling Factor Loadings — {ticker} (36-month window)")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.show()
```

### 12.3 Interpreting the Output

**Example output for AVUV (Avantis US Small Cap Value):**

```
                 AVUV    VTI
Mkt-RF          1.12   1.01    ← Both heavily loaded on market (long-only funds)
SMB             0.73   0.04    ← AVUV heavily tilted to small caps; VTI near zero
HML             0.52  -0.01    ← AVUV loaded on value; VTI is market-cap neutral
RMW             0.31   0.02    ← AVUV tilted toward profitable firms
CMA             0.12   0.00    ← AVUV slight conservative investment tilt
Mom             0.05  -0.01    ← AVUV near-zero momentum (thanks to momentum screen)
alpha           0.02%  0.01%   ← Near-zero alpha → factor exposure explains the return
annualized_alpha 0.24% 0.12%
```

**Reading this:**

- AVUV's high SMB and HML loadings confirm it is a small-cap value fund.
- The near-zero momentum loading reflects Avantis's momentum-based trading screen.
- The near-zero alpha confirms there is no "magic" — the return is fully explained by known factors.

**Example output for MTUM (iShares Momentum ETF):**

```
Mom             0.52    ← Strong positive momentum loading, as expected
SMB             0.08    ← Slight small-cap tilt
HML            -0.31    ← Negative value loading (momentum tends to be anti-value)
Mkt-RF          1.08
alpha          -0.10%   ← Slightly negative alpha → transaction costs drag
```

### 12.4 Rolling Regression Insights

Rolling regressions reveal:

- **Style drift:** A "value" fund whose HML loading declines over time may be drifting toward growth.
- **Momentum crashes:** MTUM's momentum loading crashes during sharp reversals (2009, 2020).
- **Factor crowding:** If many funds converge on the same factor loading, the factor may become temporarily overvalued.
- **Manager deception:** A manager claiming skill may simply have a stable positive size loading — detectable with rolling regressions.

### 12.5 Important Notes on Factor Data Geography

**Critical:** Factor model data must match the geography of the securities you are regressing.

- `gff.famaFrench5Factor()` returns **US-only** factor data.
- For international developed market funds (e.g., AVDV), use the developed-markets five-factor model from the Ken French data library.
- For emerging markets funds, use the emerging-markets factor model.
- Mismatched geography will give spurious results.

**Ken French Data Library:** <https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html>

---

## 13. Factor Investing in India

### 13.1 Indian Equity Market Context

India's equity market (NSE/BSE) is the world's 5th largest by market capitalization. Key characteristics relevant to factor investing:

- **Universe:** ~4,000 listed companies on NSE; Nifty 500 covers ~93% of total market cap.
- **Market structure:** Retail-heavy with ~90 million active demat accounts; institutional participation growing rapidly.
- **Liquidity:** Concentrated — top 100 stocks by market cap account for ~80% of daily turnover.
- **Short-selling:** Allowed in derivatives (F&O) but limited in cash market; long-short factor implementation is constrained.
- **Data quality:** Accounting data quality varies; related-party transactions and financial irregularities are more common than in developed markets.

### 13.2 Do Factors Work in India?

**Value (HML):** Strong evidence in India. Studies using BSE/NSE data show high B/M stocks outperform over 1995–2020 by 3–5% annually. However, the premium is episodic — value significantly underperformed during 2014–2019.

**Size (SMB):** Robust in India. Small-cap premium has historically been large (~4–6% annually). However, liquidity constraints mean the institutional small-cap premium is smaller than the academic one — many micro-cap stocks are untradeable at scale.

**Momentum:** Strong in India, particularly at 3–12 month horizons. Studies suggest an annualized momentum premium of 10–15% gross (before costs) — higher than US. The behavioral explanation is particularly relevant in India given retail-dominated price discovery.

**Profitability:** Emerging evidence supports a profitability premium. High ROE and high gross profitability stocks outperform, though the effect is smaller and noisier than in developed markets.

**Quality:** Consistent evidence. Quality metrics (low leverage, high coverage, earnings stability) predict outperformance. SEBI-regulated category "Quality" smart beta funds now track quality indices.

### 13.3 SEBI's Smart Beta Framework

SEBI (Securities and Exchange Board of India) has classified factor-based mutual funds:

- **Single-factor funds:** Allowed for Value, Momentum, Quality, Low Volatility, and Alpha (composite).
- **Multi-factor funds:** Must transparently disclose factor definitions and rebalancing.

**Existing Indian factor indices:**

- **Nifty 200 Momentum 30 Index:** Top 30 momentum stocks from Nifty 200.
- **Nifty 50 Value 20 Index:** Value stocks from Nifty 50 universe.
- **Nifty 100 Quality 30 Index:** Top 30 quality stocks from Nifty 100.
- **Nifty 100 Low Volatility 30 Index:** Lowest-volatility stocks from Nifty 100.
- **BSE India Quality Index, BSE Momentum Index:** Similar BSE equivalents.

**Key providers:** Nippon India Mutual Fund, ICICI Prudential, Mirae Asset, Motilal Oswal all offer smart beta / factor ETFs.

### 13.4 India-Specific Factor Considerations

**Earnings quality adjustments:** Given higher incidence of earnings manipulation in Indian small-caps, profitability factors should use cash flow-based metrics (operating cash flow / assets) rather than accrual-based earnings.

**Promoter holding as a quality signal:** Stocks with high and stable promoter holding tend to have lower information asymmetry. Some Indian factor strategies incorporate promoter holding as a signal.

**Government vs. private ownership:** PSU (public sector undertaking) stocks often trade at persistent value discounts due to governance concerns and political interference in management — an important consideration when implementing value strategies.

**India-specific momentum crash risk:** Indian markets are more prone to sudden halts, circuit breakers (5%/10%/20% daily limits), and regulatory interventions — all of which can cause sudden momentum crashes that are harder to hedge than in developed markets.

**Seasonality:** Indian markets show some evidence of April-May seasonality (financial year-end effects) in small-cap and value stocks. January effect is not as strong as US.

### 13.5 Running a Factor Regression on Indian Stocks

For India, you can use:

**Data Sources:**

- NSE/BSE daily price data via `nsepy`, `nsetools`, or direct NSE download.
- Financial data via Screener.in, Tijori Finance, or ACE Equity.
- Nifty 500 index as the market portfolio.
- 91-day T-bill rate as the risk-free rate (available from RBI).

**Building Factor Portfolios:**

- Construct SMB, HML, RMW for Indian universe using the Fama-French methodology applied to Nifty 500.
- Restrict to stocks with $> ₹500$ Cr market cap to avoid illiquid micro-caps.
- Rebalance annually for value/profitability, monthly for momentum.

**Example: Building India HML**

```python
import pandas as pd
import numpy as np

def build_india_hml(returns_df, book_to_market_df, market_cap_df, date):
    """
    Simple illustration of HML construction for Indian equities.
    
    returns_df: monthly returns, index=dates, columns=tickers
    book_to_market_df: B/M ratios (annual, refreshed every June)
    market_cap_df: market caps for size sort
    date: current rebalancing date
    """
    bm = book_to_market_df.loc[date].dropna()
    mc = market_cap_df.loc[date].dropna()
    common = bm.index.intersection(mc.index)
    bm = bm[common]
    mc = mc[common]
    
    # Size split: Big (top 50%) vs Small (bottom 50%)
    size_median = mc.median()
    big = mc[mc >= size_median].index
    small = mc[mc < size_median].index
    
    # B/M breakpoints (30th and 70th percentile of Big)
    bm_big = bm[big]
    value_thresh = bm_big.quantile(0.70)
    growth_thresh = bm_big.quantile(0.30)
    
    # Assign portfolios
    value_stocks = bm[bm >= value_thresh].index
    growth_stocks = bm[bm <= growth_thresh].index
    
    # Compute HML return for next month
    next_month = returns_df.index[returns_df.index > date][0]
    ret_value = returns_df.loc[next_month, value_stocks].mean()
    ret_growth = returns_df.loc[next_month, growth_stocks].mean()
    
    return ret_value - ret_growth  # HML return
```

---

## 14. Applying PCA and Factor Analysis in Quant Finance

### 14.1 Mapping the Applications

| Application | Technique | What It Solves |
|-------------|-----------|---------------|
| Risk model (covariance estimation) | FA / PCA | Estimate large covariance matrices reliably |
| Style analysis (factor exposure) | FA regression | Understand what drives a portfolio's returns |
| Pairs trading / statistical arbitrage | PCA residuals | Find cointegrated pairs after removing common factors |
| Portfolio construction | Factor portfolios | Build portfolios targeting specific risk exposures |
| Stress testing | Factor scenarios | Shock systematic factors and measure portfolio impact |
| Alternative data | NLP + PCA/FA | Compress high-dimensional text/sentiment signals |
| Regime detection | Rolling PCA | Detect structural breaks in factor structure |
| Options surface | PCA | Decompose implied volatility surface into level/slope/curvature |

### 14.2 Application 1: Risk Model and Covariance Estimation

Estimating a $D \times D$ covariance matrix for $D = 500$ stocks requires $\frac{500 \times 501}{2} \approx 125,000$ parameters — with only, say, 250 daily return observations (one year). This is drastically underdetermined: the sample covariance is noisy, non-invertible, and unstable.

**PCA-based risk model:**

1. Compute the sample covariance matrix $\hat{\Sigma}$ from $T$ observations.
2. PCA-decompose: $\hat{\Sigma} = U\Lambda U^\top$.
3. Retain $K$ leading principal components.
4. Approximate: $\hat{\Sigma}_K = U_K \Lambda_K U_K^\top + D$ where $D$ is diagonal residual.

This is exactly the FA decomposition: $\Sigma \approx WW^\top + \Psi$ where $W = U_K \Lambda_K^{1/2}$.

**Barra-style factor risk models (commercial):** MSCI Barra, Axioma, Bloomberg PORT all implement this approach — modeling the covariance of hundreds of stocks using 20–60 fundamental + statistical factors.

```python
import numpy as np
from sklearn.decomposition import PCA

def factor_risk_model(returns: np.ndarray, n_factors: int = 10):
    """
    Fit a PCA-based risk model.
    Returns factor loadings, factor covariance, idiosyncratic variance.
    """
    T, N = returns.shape
    
    # Standardize
    means = returns.mean(axis=0)
    excess = returns - means
    
    # PCA
    pca = PCA(n_components=n_factors)
    factor_returns = pca.fit_transform(excess)   # T × K factor return series
    loadings = pca.components_.T                 # N × K loading matrix
    
    # Factor covariance (diagonal in PCA space)
    factor_cov = np.cov(factor_returns.T)        # K × K
    
    # Idiosyncratic variance
    fitted = factor_returns @ loadings.T
    residuals = excess - fitted
    idio_var = np.var(residuals, axis=0)         # N-vector of specific variances
    
    # Full covariance reconstruction
    total_cov = loadings @ factor_cov @ loadings.T + np.diag(idio_var)
    
    return {
        "loadings": loadings,           # N × K
        "factor_cov": factor_cov,       # K × K
        "idio_var": idio_var,           # N
        "total_cov": total_cov,         # N × N (structured, invertible)
        "explained_variance": pca.explained_variance_ratio_
    }
```

**Key advantage:** The reconstructed $\hat{\Sigma}$ is always invertible (even when $T < N$) and much more stable out-of-sample than the raw sample covariance.

### 14.3 Application 2: PCA on Stock Returns — Identifying Systematic Components

Running PCA directly on a panel of stock returns reveals the structure of systematic risk:

```python
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Download S&P 500 returns (subset for illustration)
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
           "JPM", "BAC", "GS", "WFC", "C",
           "XOM", "CVX", "COP", "SLB", "HAL",
           "JNJ", "PFE", "MRK", "ABBV", "UNH"]

rets = yf.download(tickers, start="2018-01-01", auto_adjust=True)["Close"]
rets = rets.pct_change().dropna()

# Standardize
scaler = StandardScaler()
rets_std = scaler.fit_transform(rets)

# PCA
pca = PCA()
pca.fit(rets_std)

# Scree plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(range(1, 11), pca.explained_variance_ratio_[:10] * 100)
plt.xlabel("Principal Component")
plt.ylabel("% Variance Explained")
plt.title("Scree Plot")

plt.subplot(1, 2, 2)
plt.plot(np.cumsum(pca.explained_variance_ratio_[:10]) * 100, 'o-')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance (%)")
plt.title("Cumulative Variance")
plt.tight_layout()
plt.show()

# Component loadings heatmap
n_pc = 4
loadings = pd.DataFrame(
    pca.components_[:n_pc].T,
    index=tickers,
    columns=[f"PC{i+1}" for i in range(n_pc)]
)
print("\nFactor Loadings (first 4 PCs):")
print(loadings.round(3))
```

**What you typically find:**

- **PC1** (explains ~30–50% of variance): All stocks load positively → the *market factor*. This is essentially the SPY in disguise.
- **PC2** (~5–10%): Tech/growth stocks load positively, financials load negatively → *growth vs. value* factor.
- **PC3** (~3–5%): Energy stocks load positively, others near zero → *commodity/energy* factor.
- **PC4** (~2–3%): Healthcare vs. financials → *defensive vs. cyclical* factor.

The first few PCs of stock returns map closely to Fama-French factors — not a coincidence, since both are trying to capture the same underlying economic structure.

### 14.4 Application 3: Statistical Arbitrage via PCA Residuals

The core idea: use PCA to remove the systematic (common) component from stock returns, leaving pure idiosyncratic residuals. Pairs or baskets that were correlated before (due to common factor exposure) will diverge-and-revert in their residuals if they are economically linked.

**Approach:**

1. Fit PCA to a universe of stocks using a training window.
2. For each day, compute residuals: $\varepsilon_i = R_i - \hat{\beta}_i^\top F_t$ (actual return minus factor model predicted return).
3. Form pair/basket scores: spread = $\varepsilon_A - h \cdot \varepsilon_B$ for some hedge ratio $h$.
4. If the spread mean-reverts (stationarity test), trade when it deviates from zero.

```python
from sklearn.linear_model import LinearRegression
import statsmodels.tsa.stattools as ts

def compute_pca_residuals(returns: pd.DataFrame, n_factors: int = 5) -> pd.DataFrame:
    """Remove PCA factor returns from individual stock returns."""
    X = returns.values
    pca = PCA(n_components=n_factors)
    factor_scores = pca.fit_transform(X)    # T × K
    
    # Regress each stock on factor scores
    reg = LinearRegression(fit_intercept=True)
    reg.fit(factor_scores, X)
    fitted = reg.predict(factor_scores)
    
    residuals = pd.DataFrame(X - fitted, index=returns.index, columns=returns.columns)
    return residuals

def check_cointegration(res_A: pd.Series, res_B: pd.Series) -> dict:
    """Test if two residual series are cointegrated using ADF test on spread."""
    spread = res_A - res_B
    adf_stat, p_value, *_ = ts.adfuller(spread.dropna())
    return {"adf_stat": adf_stat, "p_value": p_value, "is_stationary": p_value < 0.05}
```

### 14.5 Application 4: PCA on Yield Curves and Volatility Surfaces

**Yield curve PCA:** The three-factor Nelson-Siegel decomposition is essentially a PCA result. Applying PCA to daily yield curve changes reveals:

- **PC1 (~90% variance):** Parallel shift (level) — all rates move together.
- **PC2 (~7%):** Steepening/flattening (slope) — short rates vs. long rates diverge.
- **PC3 (~2%):** Butterfly (curvature) — middle of curve moves relative to wings.

These PCA components are used directly in fixed income risk management and relative value trading.

**Implied volatility surface PCA:** Similarly, PCA on the vol surface (across strikes and maturities) reveals:

- **PC1:** Overall vol level shift (VIX-like).
- **PC2:** Vol skew changes (change in risk-reversal steepness).
- **PC3:** Term structure of vol (short-dated vs. long-dated vol).

Traders use these PCA components to construct hedges for vol surface exposure.

### 14.6 Application 5: WorldQuant Brain / Alpha Research

For systematic alpha research on platforms like WorldQuant Brain, FA and PCA are useful for:

**Constructing orthogonal alphas:** If you have 20 raw alpha signals, they are likely correlated. Running FA on them and retaining factor scores gives orthogonal alpha components — each capturing a genuinely distinct information source. This reduces overfitting in alpha combination.

**Neutralization:** Subtracting the portion of an alpha that correlates with known risk factors (via regression) isolates the "pure" alpha that is not just a beta disguise.

**Alpha combination via factor models:** Treat each raw alpha as a "return" and use FA to find the 3–5 underlying *meta-alphas* that drive all of them. The factor loadings tell you how much each meta-alpha contributes to each raw signal.

```python
def orthogonalize_alphas(alpha_matrix: np.ndarray, n_factors: int = 5) -> np.ndarray:
    """
    Given T×N matrix of raw alphas, return T×K orthogonalized factor scores.
    
    This finds the K underlying meta-alphas that drive the N raw alphas.
    """
    pca = PCA(n_components=n_factors)
    factor_scores = pca.fit_transform(alpha_matrix)  # T × K
    return factor_scores
```

### 14.7 Application 6: Regime Detection with Rolling PCA

The factor structure of equity markets changes across regimes (crisis vs. calm, risk-on vs. risk-off). Rolling PCA detects these structural breaks:

```python
def rolling_pca_variance(returns: pd.DataFrame, window: int = 60, n_pc: int = 1) -> pd.Series:
    """
    Rolling % variance explained by top n_pc components.
    
    High values → market moving in lockstep (crisis, high correlation)
    Low values → factor structure complex (calm, diversification works)
    """
    explained = []
    dates = []
    
    for i in range(window, len(returns)):
        window_data = returns.iloc[i-window:i]
        pca = PCA(n_components=n_pc)
        pca.fit(window_data)
        explained.append(pca.explained_variance_ratio_.sum())
        dates.append(returns.index[i])
    
    return pd.Series(explained, index=dates, name=f"PC1-{n_pc}_VarExplained")
```

**Interpretation:** A sudden spike in the variance explained by PC1 (all stocks becoming highly correlated) is a leading indicator of market stress — as occurred in March 2020, September 2008, and other crisis periods.

### 14.8 Application 7: AlphaVantage / Factor Construction Workflow

A complete workflow for building a factor model from scratch:

```
1. UNIVERSE DEFINITION
   ├── Select liquid universe (e.g., Nifty 500, Russell 1000)
   └── Apply minimum liquidity/market-cap filters

2. DATA COLLECTION
   ├── Price/return data (daily)
   ├── Fundamental data (quarterly): B/M, earnings, cash flows, assets
   └── Alternative data (optional): sentiment, satellite, web traffic

3. SIGNAL CONSTRUCTION
   ├── Value: Book-to-Market (B/M), E/P, CF/P
   ├── Momentum: 12-1 month price return
   ├── Quality: ROE, gross profitability, accruals
   └── Normalize each signal (rank-transform, then standardize)

4. FACTOR PORTFOLIO CONSTRUCTION
   ├── Cross-sectional sort into deciles/quintiles
   ├── Long top 30%, short bottom 30% (or long-only tilt for real portfolios)
   └── Cap-weight within long/short legs

5. RISK MODEL
   ├── Run PCA/FA on returns to estimate covariance matrix
   ├── Decompose into factor + specific risk
   └── Use for portfolio optimization (minimize factor risk per unit of alpha)

6. BACKTEST & VALIDATION
   ├── Walk-forward out-of-sample test (no lookahead bias)
   ├── Multiple hypothesis correction (t-stat > 3.0)
   └── Sensitivity analysis (different time periods, geographies)

7. LIVE MONITORING
   ├── Rolling factor regression to monitor factor drift
   ├── Track factor crowding (many strategies converging on same exposures)
   └── Monitor transaction costs and market impact
```

### 14.9 Summary: Which Technique When?

| Scenario | Best Technique | Why |
|----------|---------------|-----|
| Estimate covariance for optimization | PCA risk model (FA-based) | Reduces parameters, prevents overfitting |
| Understand what drives a portfolio | Factor regression (OLS) | Decomposes return into known factor contributions |
| Find mean-reverting pairs | PCA residuals | Removes common factor noise, exposes idiosyncratic |
| Understand yield curve / vol surface | PCA | Natural decomposition into level/slope/curvature |
| Build quantitative alpha factors | FA on raw signals | Orthogonalizes signals, finds meta-alphas |
| Detect market regimes | Rolling PCA variance | High PC1 variance = crisis; low = calm |
| Construct investable factor portfolio | Sort + long-short methodology | Most direct implementation of academic factors |
| Retail India factor investing | Smart beta ETFs (Nippon, ICICI) | Practical access, SEBI-regulated, low cost |

---

# Appendix: Key Mathematical Identities

$$\text{Factor Analysis model:} \quad x_i = Wz_i + \mu + \varepsilon_i$$

$$\text{Marginal covariance:} \quad \text{Cov}(x_i) = WW^\top + \Psi$$

$$\text{PPCA noise estimate:} \quad \sigma^{2*} = \frac{1}{D-L}\sum_{j=L+1}^D \lambda_j$$

$$\text{PCA explained variance:} \quad \text{Proportion}_k = \frac{\lambda_k}{\sum_{j=1}^p \lambda_j}$$

$$\text{Factor regression:} \quad R_i - R_f = \alpha_i + \sum_{k=1}^K \beta_{ik} F_{k} + \varepsilon_i$$

$$\text{CAPM:} \quad \mathbb{E}[R_i] = R_f + \beta_i(\mathbb{E}[R_m] - R_f)$$

$$\text{Communality:} \quad h_d^2 = \sum_{k=1}^L f_{dk}^2, \quad \text{Uniqueness:} \quad u_d = 1 - h_d^2$$

$$\text{Annualized alpha from monthly:} \quad \alpha_{\text{annual}} = (1 + \alpha_{\text{monthly}})^{12} - 1$$

---

# Key References

1. **Fama, E.F. & French, K.R. (1993).** Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56.
2. **Fama, E.F. & French, K.R. (2015).** A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1–22.
3. **Carhart, M.M. (1997).** On persistence in mutual fund performance. *Journal of Finance*, 52(1), 57–82.
4. **Ross, S.A. (1976).** The arbitrage theory of capital asset pricing. *Journal of Economic Theory*, 13(3), 341–360.
5. **Sharpe, W.F. (1991).** The arithmetic of active management. *Financial Analysts Journal*, 47(1), 7–9.
6. **Asness, C., Frazzini, A., & Pedersen, L.H. (2019).** Quality minus junk. *Review of Accounting Studies*, 24(1), 34–112.
7. **Frazzini, A., Kabiller, D., & Pedersen, L.H. (2018).** Buffett's alpha. *Financial Analysts Journal*, 74(4), 35–55.
8. **Harvey, C.R., Liu, Y., & Zhu, H. (2016).** ... and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68.
9. **Tipping, M.E. & Bishop, C.M. (1999).** Probabilistic principal component analysis. *Journal of the Royal Statistical Society: Series B*, 61(3), 611–622.
10. **Ang, A. (2014).** *Asset Management: A Systematic Approach to Factor Investing*. Oxford University Press.
11. **Katchova, A. (2013).** Principal Component Analysis and Factor Analysis Lecture Notes.

---
