# Volatility Clustering in U.S. Technology: Software vs. Semiconductors

## A Comparative Study (2020–2025)

### Research Question

> **How does volatility clustering differ between software and semiconductor
> companies in the U.S. technology sector over the past five years (2020–2025)?**

**Research Conductor:** Mariana Andrade
**Research Advisor:** Dr. Sandstrom

---

## Overview

This study compares **volatility clustering** — the tendency for large price
movements to be followed by large movements, and calm by calm — between two
industries inside the U.S. technology sector: **software** and **semiconductors**.

Using daily returns from 2020–2025, the analysis fits **GARCH(1,1)** models to
each group and measures how persistent volatility shocks are (the α + β
persistence parameter), supported by the autocorrelation of squared returns.
The goal is a quantified answer to which industry's volatility shocks last
longer, and an interpretation of why.

---

## Expected Outcome & Deliverables

**Outcome:** A quantified comparison of volatility clustering between U.S.
software and semiconductor firms (2020–2025), using GARCH persistence and
return autocorrelation, showing which industry's volatility shocks are more
persistent and why.

**Deliverables:**
1. A clean volatility **panel** (daily log returns + monthly realized volatility) for both groups
2. **Stationarity results** (ADF + KPSS) justifying the use of returns
3. **GARCH(1,1) models** per group, reporting α + β persistence
4. **Figures:** rolling volatility (software vs. semiconductors), ACF of squared returns, persistence comparison
5. **Written findings** (3–5 pages): question, data, method, results, interpretation, limitations

---

## Data Source

Data is pulled live from the **yfinance Sector / Industry API**
(not a static dataset), using the technology-sector industry keys:

| Group         | yfinance industry keys                              |
| ------------- | --------------------------------------------------- |
| Software      | `software-application`, `software-infrastructure`   |
| Semiconductor | `semiconductors`, `semiconductor-equipment-materials` |

For each industry, the top companies are retrieved via
`yfinance.Industry(key).top_companies`, and daily adjusted prices are
downloaded with `yfinance.download` for the period **2020-01-01 to 2025-12-31**.

---

## Methodology

1. **Returns:** daily log returns from adjusted close prices.
2. **Stationarity:** Augmented Dickey-Fuller (ADF) and KPSS tests on the return
   series of each group. ADF null = non-stationary; KPSS null = stationary.
   Agreement confirms returns are stationary and suitable for modeling.
3. **Volatility clustering (EDA):** rolling realized volatility and the
   autocorrelation function (ACF) of squared returns — the visual signature of
   clustering.
4. **GARCH(1,1):** fit per group; the persistence parameter **α + β** is the
   core clustering measure. Higher persistence = shocks fade more slowly.
5. **Comparison:** software vs. semiconductor persistence, with interpretation.

---

## Contact & Support

For questions or discussions about this research, please open an issue in the repository.
