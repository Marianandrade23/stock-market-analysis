"""
Initial GARCH(1,1) model estimation — Software vs. Semiconductor volatility clustering
Research question: How does volatility clustering differ between software and
semiconductor companies in the U.S. tech sector, 2020-2025?

Pipeline:
  1. Pull top companies per industry using yfinance.Industry (NOT yfinance.Sector —
     Sector aggregates across the whole "technology" sector; Industry scopes to
     one industry key, which is what we want for the software/semiconductor split).
  2. Build daily log-return panels for each group.
  3. EDA: rolling volatility + ACF of squared returns (visual evidence of clustering).
  4. Fit GARCH(1,1) per ticker, then compare average persistence (alpha + beta)
     and volatility half-life between the two groups.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf

START = "2020-01-01"
END = "2025-12-31"
N_PER_INDUSTRY = 20  # top N companies pulled per industry key, before capping group size

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


# ---------------------------------------------------------------------------
# 4. GARCH(1,1) estimation per ticker, aggregated to group level
# ---------------------------------------------------------------------------
def fit_garch_per_ticker(returns, group_name):
    """Fit GARCH(1,1) to each ticker's return series; return summary DataFrame."""
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
            })
        except Exception as e:
            print(f"  [warn] GARCH failed for {ticker}: {e}")
    return pd.DataFrame(records)


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

    print("\nStep 3: EDA — rolling volatility and ACF of squared returns...")
    plot_rolling_volatility(returns_by_group)
    plot_squared_returns_acf(returns_by_group)

    print("\nStep 4: Fitting GARCH(1,1) per ticker...")
    all_results = []
    for group_name, returns in returns_by_group.items():
        res_df = fit_garch_per_ticker(returns, group_name)
        all_results.append(res_df)
    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv("garch_results_by_ticker.csv", index=False)
    print("Saved: garch_results_by_ticker.csv")

    print("\nGroup-level summary (persistence = alpha + beta):")
    summary = summarize_group_results(results_df)
    print(summary)
    summary.to_csv("garch_group_summary.csv")
    print("Saved: garch_group_summary.csv")
