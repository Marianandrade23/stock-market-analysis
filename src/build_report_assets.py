"""
Build report-ready figures and tables from existing GARCH results.
Addresses advisor feedback from 7/17/26:
  - Hide t=0 in ACF plots (scaling issue caused by the always-1.0 spike at lag 0)
  - Show company name alongside ticker
  - Facet-wrap ALL tickers for an appendix meta-overview (not just selected ones)
  - Visually highlight volatility-clustering regions in log-return plots
  - Well-formatted, rounded, labeled tables ready for a doc/slide deck

Run this AFTER garch_initial_model.py (it reuses that script's saved outputs).
Place in src/ alongside garch_initial_model.py.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
from statsmodels.graphics.tsaplots import plot_acf

# Re-use the same config as the main script
from garch_initial_model import (
    START, END, INDUSTRY_GROUPS, build_ticker_groups, build_returns_panel
)

RESULTS_TICKER_CSV = "results/tables/garch_results_by_ticker.csv"
RESULTS_SUMMARY_CSV = "results/tables/garch_group_summary.csv"


# ---------------------------------------------------------------------------
# Ticker -> company name mapping
# ---------------------------------------------------------------------------
def build_name_map(tickers):
    """Map ticker symbols to short company names for readable plot titles."""
    name_map = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            name_map[t] = info.get("shortName", t)
        except Exception:
            name_map[t] = t
    return name_map


def label(ticker, name_map, max_len=22):
    """'AAPL' -> 'AAPL — Apple Inc.' (truncated if long)."""
    name = name_map.get(ticker, ticker)
    if len(name) > max_len:
        name = name[:max_len - 1] + "…"
    return f"{ticker} — {name}" if name != ticker else ticker


# ---------------------------------------------------------------------------
# Log-return plots with clustering visually highlighted
# ---------------------------------------------------------------------------
def plot_log_returns_highlighted(returns, tickers, name_map, group_name,
                                   std_multiple=1.0, out_path=None):
    """Plot raw log returns with a shaded +/- N*std band; points outside the
    band (i.e. the volatility-clustering bursts) are colored differently so
    a general audience can see clustering at a glance."""
    n = len(tickers)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, ticker in zip(axes, tickers):
        series = returns[ticker].dropna()
        std = series.std()
        band = std_multiple * std
        colors = np.where(series.abs() > band, "#d62728", "#1f77b4")  # red = outside band
        ax.scatter(series.index, series, c=colors, s=4, alpha=0.7)
        ax.axhspan(-band, band, color="gray", alpha=0.15, label=f"±{std_multiple:.0f} std dev")
        ax.set_title(label(ticker, name_map, 30), fontsize=10, loc="left")
        ax.set_ylabel("log return")
    axes[-1].legend(loc="upper right", fontsize=8)
    red_patch = mpatches.Patch(color="#d62728", label="Outside normal range (clustering)")
    fig.legend(handles=[red_patch], loc="upper right", fontsize=8, bbox_to_anchor=(0.98, 0.995))
    fig.suptitle(f"Log Returns — {group_name.title()} (red = outside ±{std_multiple:.0f} std dev band)")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")
    return fig


def plot_log_returns_facet_all(returns, name_map, group_name, out_path, ncols=5):
    """Small-multiples grid of ALL tickers' log returns, for appendix use."""
    tickers = list(returns.columns)
    nrows = int(np.ceil(len(tickers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.2 * nrows), sharex=True)
    axes = axes.flatten()
    for ax, ticker in zip(axes, tickers):
        series = returns[ticker].dropna()
        ax.plot(series.index, series, linewidth=0.4)
        ax.set_title(ticker, fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes[len(tickers):]:
        ax.axis("off")
    fig.suptitle(f"All {group_name.title()} Tickers — Log Returns (Appendix)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# ACF plots — hide t=0, add company names, facet-wrap all tickers
# ---------------------------------------------------------------------------
def plot_squared_returns_acf_fixed(returns_dict, lags=30, out_path="results/figures/acf_squared_returns.png"):
    """Same as original ACF plot but with zero=False to hide the always-1.0
    spike at lag 0, which was compressing the scale of the real bars."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (group_name, returns) in zip(axes, returns_dict.items()):
        group_mean_return = returns.mean(axis=1)
        squared = group_mean_return.dropna() ** 2
        plot_acf(squared, lags=lags, ax=ax, zero=False,
                 title=f"ACF of squared returns — {group_name.title()}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_residual_acf_facet_all(fitted_models, name_map, group_name, out_path, ncols=5, lags=20):
    """Small-multiples grid of residual ACFs for ALL tickers, zero excluded."""
    tickers = list(fitted_models.keys())
    nrows = int(np.ceil(len(tickers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows))
    axes = axes.flatten()
    for ax, ticker in zip(axes, tickers):
        std_resid = fitted_models[ticker].std_resid.dropna()
        plot_acf(std_resid, lags=lags, ax=ax, zero=False, title=ticker)
        ax.tick_params(labelsize=6)
        ax.title.set_size(8)
    for ax in axes[len(tickers):]:
        ax.axis("off")
    fig.suptitle(f"All {group_name.title()} Tickers — Residual ACF (Appendix)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Well-formatted tables
# ---------------------------------------------------------------------------
def format_ticker_table(results_df, name_map, out_csv, out_md):
    df = results_df.copy()
    df["Company"] = df["ticker"].map(name_map).fillna(df["ticker"])
    df = df.rename(columns={
        "ticker": "Ticker", "group": "Group", "alpha": "α (ARCH)", "beta": "β (GARCH)",
        "persistence": "Persistence (α+β)", "half_life_days": "Half-Life (days)",
        "lb_pvalue_resid": "Ljung-Box p (resid)", "lb_pvalue_resid_sq": "Ljung-Box p (resid²)",
        "aic": "AIC",
    })
    cols = ["Ticker", "Company", "Group", "α (ARCH)", "β (GARCH)", "Persistence (α+β)",
            "Half-Life (days)", "Ljung-Box p (resid)", "Ljung-Box p (resid²)", "AIC"]
    df = df[cols]
    round_map = {c: 3 for c in cols if c not in ("Ticker", "Company", "Group")}
    round_map["Half-Life (days)"] = 1
    df = df.round(round_map)
    df.to_csv(out_csv, index=False)
    df.to_markdown(out_md, index=False)
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")
    return df


def format_summary_table(summary_df, out_csv, out_md):
    df = summary_df.copy()
    df = df.round(3)
    df.to_csv(out_csv)
    with open(out_md, "w") as f:
        f.write(df.to_markdown())
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading existing GARCH results...")
    results_df = pd.read_csv(RESULTS_TICKER_CSV)
    summary_df = pd.read_csv(RESULTS_SUMMARY_CSV, header=[0, 1], index_col=0)

    print("\nRebuilding ticker groups and return panels (for return-level plots)...")
    ticker_groups = build_ticker_groups()
    returns_by_group = {g: build_returns_panel(t) for g, t in ticker_groups.items()}

    print("\nBuilding name map (this hits Yahoo Finance once per ticker, may take a bit)...")
    all_tickers = [t for tickers in ticker_groups.values() for t in tickers]
    name_map = build_name_map(all_tickers)
    pd.Series(name_map, name="company_name").to_csv("results/tables/ticker_name_map.csv")

    print("\nBuilding highlighted log-return plots for select tickers...")
    for group_name, returns in returns_by_group.items():
        chosen = returns.count().sort_values(ascending=False).index[:3]
        plot_log_returns_highlighted(
            returns, chosen, name_map, group_name,
            out_path=f"results/figures/log_returns_highlighted_{group_name}.png"
        )

    print("\nBuilding appendix facet-wrap of ALL tickers (log returns)...")
    for group_name, returns in returns_by_group.items():
        plot_log_returns_facet_all(
            returns, name_map, group_name,
            out_path=f"results/figures/appendix_log_returns_all_{group_name}.png"
        )

    print("\nRebuilding ACF of squared returns with t=0 hidden...")
    plot_squared_returns_acf_fixed(returns_by_group)

    print("\nFormatting tables (rounded, labeled)...")
    format_ticker_table(results_df, name_map,
                         "results/tables/garch_results_formatted.csv",
                         "results/tables/garch_results_formatted.md")
    format_summary_table(summary_df,
                          "results/tables/garch_group_summary_formatted.csv",
                          "results/tables/garch_group_summary_formatted.md")

    print("\nDone. Note: residual facet-wrap for ALL tickers requires re-fitting GARCH")
    print("models (fitted objects aren't saved to disk). Run that separately if needed —")
    print("ask Claude for a version that refits and saves models via pickle if you want it.")
