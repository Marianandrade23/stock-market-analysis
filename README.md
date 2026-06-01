# Stock Market Trend Analysis and Price Prediction Using Data Analytics
## A Study of US Market Sectors (2025–2026)

### Research Overview

This research applies **data analytics techniques** to examine stock market behavior across major US industries using the **US Stock Market Historical OHLCV Dataset** (120 companies, 9 sectors, 184K+ records).

**Research Conductor:** Mariana Andrade  
**Research Advisor:** Dr. Sandstrom

---

## Research Focus & Objectives

### Primary Analysis Areas:

1. **Volume Patterns & Market Activity**
   - Trading volume analysis and market participation levels
   - Volume-price relationships and correlations
   - Liquidity indicators by sector
   - Volume spikes and anomalies

2. **Volatility Analysis**
   - Volatility clustering and persistence
   - GARCH models to capture time-varying volatility
   - Volatility regimes (low, normal, high volatility periods)
   - Volatility forecasting

3. **Sector-Specific Volatility Impact**
   - How volume and volatility differ across 9 sectors
   - Sector risk profiles and volatility rankings
   - Identification of stable vs. volatile sectors
   - Cross-sector volatility comparisons

4. **Volume-Volatility Relationship**
   - Does trading volume predict volatility?
   - How do volume spikes affect market risk?
   - Correlation between volume patterns and volatility clustering
   - Sector differences in volume-volatility dynamics

5. **Visualization & Communication**
   - Interactive charts and dashboards
   - Volume and volatility metrics by sector
   - Time series decomposition visualizations
   - Risk profile dashboards for decision-makers

---

## Data Source

**Dataset:** [US Stock Market Historical OHLCV Dataset](https://www.kaggle.com/datasets/asadullahcreative/us-stock-market-historical-ohlcv-dataset)

- **Coverage:** 120 companies
- **Sectors:** 9 major US industries
- **Records:** 184,000+
- **Period:** Historical OHLCV data
- **Columns:** Open, High, Low, Close, Volume

---

## Theoretical Framework

### Volume Analysis
- **On-Balance Volume (OBV):** Cumulative volume indicator
- **Volume Rate of Change (VROC):** Volume momentum
- **Price-Volume Trend:** Combined price and volume indicator
- **Volume Weighted Average Price (VWAP):** Price adjusted by volume

### Volatility Modeling
- **GARCH Models:** Time-varying conditional volatility
- **Volatility Regimes:** Low, Normal, High volatility states
- **Volatility Clustering:** Periods of high volatility tend to persist
- **Rolling Volatility:** Time-window based volatility estimation
- **Annualized Volatility:** Sector and cross-sector comparisons

### Volume-Volatility Relationship
- **Empirical Evidence:** Volume often predicts volatility
- **Microstructure Theory:** Trading activity affects price discovery
- **Information Flow:** Volume as proxy for information arrival
- **Liquidity Risk:** How volume affects market depth

### Financial Risk Metrics
- **Value at Risk (VaR):** Maximum potential loss under normal conditions
- **Stock Returns Analysis:** Log returns, cumulative returns, risk-adjusted returns
- **Sharpe Ratio:** Risk-adjusted return by sector
- **Drawdown Analysis:** Maximum loss from peak

---

## Project Structure

```
stock-market-analysis/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── data/
│   ├── raw/                          # Original KAGGLE dataset
│   ├── processed/                    # Cleaned and preprocessed data
│   └── README_DATA.md                # Data dictionary and loading instructions
├── notebooks/
│   ├── 01_Data_Loading_EDA.ipynb     # Initial data exploration
│   ├── 02_Volume_Pattern_Analysis.ipynb # Trading volume insights and OBV
│   ├── 03_Volatility_Analysis.ipynb  # Volatility clustering, GARCH models
│   ├── 04_Sector_Comparison.ipynb    # Volume & volatility by sector
│   ├── 05_Volume_Volatility_Relationship.ipynb # Correlation & causality
│   ├── 06_Risk_Metrics_by_Sector.ipynb # VaR, Sharpe, drawdowns
│   ├── 07_Time_Series_Models.ipynb   # GARCH, forecasting
│   └── 08_Interactive_Dashboard.ipynb # Comprehensive sector dashboard
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Yahoo Finance API integration
│   ├── preprocessing.py              # Data cleaning and transformation
│   ├── volume_analysis.py            # OBV, VROC, VWAP calculations
│   ├── volatility_analysis.py        # GARCH, rolling volatility
│   ├── sector_analysis.py            # Sector grouping and comparison
│   ├── visualization.py              # Plotting and dashboards
│   └── risk_metrics.py               # VaR, Sharpe ratio calculations
├── results/
│   ├── figures/                      # Generated charts and plots
│   ├── reports/                      # Analysis reports
│   └── dashboards/                   # Interactive dashboards (Plotly)
├── literature/
│   ├── references.md                 # Key papers and textbooks
│   ├── volume_theory.md              # Volume analysis concepts
│   ├── volatility_models.md          # GARCH and volatility theory
│   └── sector_profiles.md            # Sector-specific insights
├── ROADMAP.md                        # Project timeline and milestones
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Libraries: pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn, yfinance, arch (GARCH)

### Installation

```bash
# Clone the repository
git clone https://github.com/Marianandrade23/stock-market-analysis.git
cd stock-market-analysis

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Quick Start

1. **Download Dataset:** Obtain the OHLCV dataset from Kaggle and place in `data/raw/`
2. **Run Notebooks:** Start with `01_Data_Loading_EDA.ipynb`
3. **Analyze Volume:** Use `02_Volume_Pattern_Analysis.ipynb`
4. **Study Volatility:** Use `03_Volatility_Analysis.ipynb`
5. **Compare Sectors:** Use `04_Sector_Comparison.ipynb`
6. **Explore Relationships:** Use `05_Volume_Volatility_Relationship.ipynb`

---

## Key Research Questions

**Volume Patterns:**
- How does trading volume vary across sectors and time?
- What patterns emerge during market anomalies?
- How does volume relate to price movements?

**Volatility Dynamics:**
- Which sectors exhibit the highest volatility?
- Is volatility persistent (clustering)?
- Can GARCH models capture sector-specific volatility patterns?

**Cross-Sector Impact:**
- How do volume and volatility differ across the 9 sectors?
- Are tech stocks more volatile than energy stocks?
- Do different sectors respond differently to market events?

**Volume-Volatility Relationship:**
- Does high trading volume predict increased volatility?
- Is this relationship consistent across all sectors?
- Can we use volume to forecast volatility?

**Risk Implications:**
- Which sectors carry the highest risk (VaR)?
- How do volume patterns affect risk metrics?
- What are the risk-adjusted returns by sector?

---

## Tools & Technologies

- **Python:** pandas, numpy, scipy, statsmodels, scikit-learn
- **Volatility Modeling:** arch package (GARCH)
- **Visualization:** matplotlib, seaborn, plotly
- **Data Sources:** Yahoo Finance API (yfinance), Kaggle dataset
- **Statistical Analysis:** Volume indicators, GARCH models, correlation analysis
- **Dashboards:** Plotly, Jupyter interactive widgets

---

## Expected Outputs

1. **Volume Analysis Report:** Trading patterns and OBV metrics by sector
2. **Volatility Study:** GARCH models, volatility clustering, regimes by sector
3. **Sector Comparison:** Volume and volatility rankings and profiles
4. **Volume-Volatility Report:** Correlation analysis, predictive relationships
5. **Risk Metrics:** VaR, Sharpe ratios, drawdowns by sector
6. **Interactive Dashboard:** Comprehensive visualization of volume, volatility, and risk
7. **Research Findings:** Key conclusions on how volume and volatility affect different sectors

---

## Next Steps

- [ ] Download and load Kaggle dataset
- [ ] Set up Yahoo Finance API integration
- [ ] Perform initial EDA and data quality checks
- [ ] Calculate volume indicators (OBV, VROC, VWAP)
- [ ] Develop GARCH volatility models
- [ ] Compare volume and volatility across sectors
- [ ] Analyze volume-volatility relationships
- [ ] Create interactive dashboards
- [ ] Document findings and conclusions

---

## References & Literature

**Key Topics:**
- Volume analysis and trading patterns
- GARCH models and volatility clustering
- Volume-volatility relationships in financial markets
- Sector-specific market microstructure
- Risk metrics and Value at Risk
- Asset pricing and sector performance

See `literature/references.md` for detailed citations and reading materials.

---

## Author

**Mariana Andrade**  
Research conducted under guidance of **Dr. Sandstrom**

---

## License

This project is for educational and research purposes.

---

## Contact & Support

For questions or discussions about this research, please open an issue in the repository.
