"""
Initial GARCH(1,1) model estimation — Software vs. Semiconductor volatility clustering
Research question: How does volatility clustering differ between software and
semiconductor companies in the U.S. tech sector, 2020-2025?

Pipeline:
  1. Pull top companies per industry using yfinance.Industry (NOT yfinance.Sector —
     Sector aggregates across the whole "technology" sector; Industry scopes to
     one industry key, which is what we want for the software/semiconductor split).
  2. Build daily log-return panels for each group.
  3. EDA: raw log-return plots for representative tickers, rolling volatility,
     and ACF of squared returns (visual evidence of clustering).
  4. Fit GARCH(1,1) per ticker, then compare average persistence (alpha + beta)
     and volatility half-life between the two groups.
  5. Residual diagnostics: check whether standardized GARCH residuals are white
     noise via Ljung-Box on residuals and squared residuals.

Advisor notes addressed (7/9/26):
  1. Returns metric confirmed as log returns: np.log(data / data.shift(1)).
  2. Added raw log-return plots for select tickers (see plot_log_returns_selected).
  3. Added residual white-noise checks (see run_residual_diagnostics).
  4. Repo cleanup handled separately (venv/.vs removed from tracking).
  5. N_PER_INDUSTRY bumped from 20 -> 50 to test robustness of results.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

START = "2020-01-01"
END = "2025-12-31"
N_PER_INDUSTRY = 50  # bumped from 20 per advisor's robustness question

INDUSTRY_GROUPS = {
    "software": ["software-application", "software-infrastructure"],
    "semiconductor": ["semiconductors", "semiconductor-equipment-materials"],
}


# ---------------------------------------------------------------------------
# 1. Get tickers per group via yfinance.Industry
# ---------------------------------------------------------------------------
def get_industry_tickers(industry_key, n=N_PER_INDUSTRY):
    """Return top-N ticker symbols for a given yfinance industry key."""
    try:
        industry = yf.Industry(industry_key)
        top = industry.top_companies  # DataFrame indexed by symbol
        if top is None or top.empty:
            print(f"  [warn] no top_companies returned for '{industry_key}'")
            return []
        return list(top.index[:n])
    except Exception as e:
        print(f"  [warn] failed to fetch '{industry_key}': {e}")
        return []


def build_ticker_groups():
    groups = {}
    for group_name, keys in INDUSTRY_GROUPS.items():
        tickers = []
        for key in keys:
            found = get_industry_tickers(key)
            print(f"{group_name} / {key}: {len(found)} tickers")
            tickers.extend(found)
        # de-dupe, preserve order
        groups[group_name] = list(dict.fromkeys(tickers))
    return groups


# ---------------------------------------------------------------------------
# 2. Build daily log-return panel
# ---------------------------------------------------------------------------
def build_returns_panel(tickers, start=START, end=END):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):  # single ticker edge case
        data = data.to_frame()
    log_returns = np.log(data / data.shift(1)).dropna(how="all")
    return log_returns


# ---------------------------------------------------------------------------
# 3. EDA — rolling volatility + ACF of squared returns
# ---------------------------------------------------------------------------
def plot_rolling_volatility(returns_dict, window=21, out_path="rolling_volatility.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    for group_name, returns in returns_dict.items():
        group_mean_return = returns.mean(axis=1)  # equal-weighted group series
        rolling_vol = group_mean_return.rolling(window).std() * np.sqrt(252)
        ax.plot(rolling_vol.index, rolling_vol, label=f"{group_name} (annualized {window}d rolling vol)")
    ax.set_title("Rolling Volatility: Software vs. Semiconductor (2020-2025)")
    ax.set_ylabel("Annualized volatility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_squared_returns_acf(returns_dict, lags=30, out_path="acf_squared_returns.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (group_name, returns) in zip(axes, returns_dict.items()):
        group_mean_return = returns.mean(axis=1)
        squared = group_mean_return.dropna() ** 2
        plot_acf(squared, lags=lags, ax=ax, title=f"ACF of squared returns — {group_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_log_returns_selected(returns_dict, n_tickers=3, out_path="log_returns_selected.png"):
    """Plot raw log returns (not rolling vol) for a few representative tickers per
    group, so the clustering pattern is visible before any model is fit."""
    fig, axes = plt.subplots(2, n_tickers, figsize=(5 * n_tickers, 7), sharex=True)
    for row, (group_name, returns) in enumerate(returns_dict.items()):
        # pick tickers with the most complete history as "representative"
        chosen = returns.count().sort_values(ascending=False).index[:n_tickers]
        for col, ticker in enumerate(chosen):
            ax = axes[row, col]
            series = returns[ticker].dropna()
            ax.plot(series.index, series, linewidth=0.6)
            ax.set_title(f"{group_name}: {ticker}", fontsize=10)
            ax.set_ylabel("log return")
    fig.suptitle("Raw Log Returns — Representative Tickers (clustering visible as volatility bursts)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# 4. GARCH(1,1) estimation per ticker, aggregated to group level
# ---------------------------------------------------------------------------
def fit_garch_per_ticker(returns, group_name, fitted_models=None):
    """Fit GARCH(1,1) to each ticker's return series; return summary DataFrame.
    Includes Ljung-Box p-values on standardized residuals (and squared residuals)
    as a white-noise diagnostic. If fitted_models dict is passed, stores the
    fitted result object per ticker for later plotting."""
    records = []
    for ticker in returns.columns:
        series = returns[ticker].dropna() * 100  # scale to % for numerical stability
        if len(series) < 250:  # skip tickers with too little history
            continue
        try:
            model = arch_model(series, vol="Garch", p=1, q=1, dist="t")
            res = model.fit(disp="off")
            alpha = res.params.get("alpha[1]", np.nan)
            beta = res.params.get("beta[1]", np.nan)
            persistence = alpha + beta
            half_life = np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.nan

            # Residual diagnostics: standardized residuals should be white noise
            # if the model is well-specified.
            std_resid = res.std_resid.dropna()
            lb_resid = acorr_ljungbox(std_resid, lags=[10], return_df=True)
            lb_resid_sq = acorr_ljungbox(std_resid ** 2, lags=[10], return_df=True)
            lb_resid_pvalue = lb_resid["lb_pvalue"].iloc[0]
            lb_resid_sq_pvalue = lb_resid_sq["lb_pvalue"].iloc[0]

            records.append({
                "group": group_name,
                "ticker": ticker,
                "omega": res.params.get("omega", np.nan),
                "alpha": alpha,
                "beta": beta,
                "persistence": persistence,
                "half_life_days": half_life,
                "log_likelihood": res.loglikelihood,
                "aic": res.aic,
                "lb_pvalue_resid": lb_resid_pvalue,          # >0.05 -> residuals ~ white noise
                "lb_pvalue_resid_sq": lb_resid_sq_pvalue,    # >0.05 -> no leftover ARCH effect
            })
            if fitted_models is not None:
                fitted_models[ticker] = res
        except Exception as e:
            print(f"  [warn] GARCH failed for {ticker}: {e}")
    return pd.DataFrame(records)


def plot_residual_diagnostics(fitted_models, tickers, out_path="garch_residual_diagnostics.png"):
    """Plot standardized residuals + their ACF for a handful of tickers, to
    visually confirm whether GARCH residuals look like white noise."""
    n = len(tickers)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.2 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    for row, ticker in enumerate(tickers):
        if ticker not in fitted_models:
            continue
        std_resid = fitted_models[ticker].std_resid.dropna()
        axes[row, 0].plot(std_resid.index, std_resid, linewidth=0.6)
        axes[row, 0].set_title(f"{ticker}: standardized residuals")
        plot_acf(std_resid, lags=20, ax=axes[row, 1], title=f"{ticker}: ACF of residuals")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def summarize_group_results(results_df):
    summary = results_df.groupby("group")[["alpha", "beta", "persistence", "half_life_days"]].agg(
        ["mean", "median", "std"]
    )
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Step 1: Pulling industry ticker lists...")
    ticker_groups = build_ticker_groups()

    print("\nStep 2: Building return panels...")
    returns_by_group = {}
    for group_name, tickers in ticker_groups.items():
        print(f"  {group_name}: {len(tickers)} tickers -> downloading price history")
        returns_by_group[group_name] = build_returns_panel(tickers)

    print("\nStep 3: EDA — raw log returns, rolling volatility, ACF of squared returns...")
    plot_log_returns_selected(returns_by_group)
    plot_rolling_volatility(returns_by_group)
    plot_squared_returns_acf(returns_by_group)

    print("\nStep 4: Fitting GARCH(1,1) per ticker...")
    all_results = []
    fitted_models = {}  # ticker -> fitted arch result, kept for residual plots
    for group_name, returns in returns_by_group.items():
        res_df = fit_garch_per_ticker(returns, group_name, fitted_models=fitted_models)
        all_results.append(res_df)
    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv("garch_results_by_ticker.csv", index=False)
    print("Saved: garch_results_by_ticker.csv")

    print("\nGroup-level summary (persistence = alpha + beta):")
    summary = summarize_group_results(results_df)
    print(summary)
    summary.to_csv("garch_group_summary.csv")
    print("Saved: garch_group_summary.csv")

    print("\nStep 5: Residual diagnostics (white-noise check)...")
    pct_white_noise = (results_df["lb_pvalue_resid"] > 0.05).mean() * 100
    pct_no_arch_left = (results_df["lb_pvalue_resid_sq"] > 0.05).mean() * 100
    print(f"  {pct_white_noise:.1f}% of tickers have white-noise residuals (Ljung-Box p > 0.05)")
    print(f"  {pct_no_arch_left:.1f}% of tickers show no remaining ARCH effect in squared residuals")

    # Plot residual diagnostics for a couple of representative tickers per group
    sample_tickers = []
    for group_name, returns in returns_by_group.items():
        group_tickers = results_df.loc[results_df["group"] == group_name, "ticker"]
        sample_tickers.extend(group_tickers.head(2).tolist())
    plot_residual_diagnostics(fitted_models, sample_tickers)