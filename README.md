# Stock Market Trend Analysis and Price Prediction Using Data Analytics
## A Study of US Market Sectors (2025–2026)

### Research Overview

This research applies **data analytics techniques** to examine stock market behavior across major US industries using the **US Stock Market Historical OHLCV Dataset** (120 companies, 9 sectors, 184K+ records).

**Research Conductor:** Dr. Sandstrom

---

## Research Focus & Objectives

### Primary Analysis Areas:

1. **Price Trends**
   - How Open, High, Low, Close (OHLC) prices move over time
   - Trend identification and pattern recognition
   - Sector-specific price movements

2. **Volume Patterns**
   - Trading volume analysis and what it reveals about market activity
   - Volume-price relationships
   - Liquidity indicators by sector

3. **Sector Comparison**
   - Volatility analysis across major US industries (Tech, Healthcare, Energy, Finance, etc.)
   - Identification of stable vs. volatile sectors
   - Sector performance benchmarking

4. **Descriptive & Exploratory Data Analysis (EDA)**
   - Statistical summaries of market data
   - Outlier detection and analysis
   - Market close patterns and anomalies

5. **Visualization & Communication**
   - Interactive charts and dashboards
   - Clear visual communication of findings
   - Insights for investors, analysts, and decision-makers

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

### Time Series Analysis
- **Stationarity Testing:** ADF (Augmented Dickey-Fuller) tests
- **Autocorrelation Analysis:** ACF/PACF plots
- **Autoregressive Models:** AR, ARIMA, ARIMAX
- **Time Series Decomposition:** Trend, Seasonality, Residuals

### Financial Risk Metrics
- **Volatility Clustering:** GARCH models, volatility regimes
- **Value at Risk (VaR):** Parametric and non-parametric approaches
- **Stock Returns Analysis:** Log returns, cumulative returns, risk-adjusted returns

### Statistical Literature
- Time series fundamentals and autocorrelation theory
- Asset pricing models
- Return distributions and tail behavior
- Market microstructure

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
│   ├── 02_Price_Trend_Analysis.ipynb # OHLC analysis and trends
│   ├── 03_Volume_Pattern_Analysis.ipynb # Trading volume insights
│   ├── 04_Sector_Comparison.ipynb    # Sector-level analysis
│   ├── 05_Volatility_Analysis.ipynb  # Volatility clustering, GARCH
│   ├── 06_Time_Series_Analysis.ipynb # Stationarity, ACF/PACF, ARIMA
│   ├── 07_VaR_and_Risk_Metrics.ipynb # Value at Risk calculations
│   └── 08_Outlier_Detection.ipynb    # Market anomalies and outliers
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Yahoo Finance API integration
│   ├── preprocessing.py              # Data cleaning and transformation
│   ├── analysis.py                   # Statistical analysis functions
│   ├── visualization.py              # Plotting and dashboards
│   └── time_series_models.py         # ARIMA, GARCH implementations
├── results/
│   ├── figures/                      # Generated charts and plots
│   ├── reports/                      # Analysis reports
│   └── dashboards/                   # Interactive dashboards (Power BI, Plotly)
├── literature/
│   ├── references.md                 # Key papers and textbooks
│   ├── chapter_notes.md              # Notes from time series literature
│   └── models_summary.md             # Model descriptions and formulas
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Libraries: pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn, yfinance

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
3. **Explore Sectors:** Use `04_Sector_Comparison.ipynb` for sector-level insights
4. **Advanced Analysis:** Progress to time series and risk analysis notebooks

---

## Key Research Questions

- How do price trends and volatility vary across sectors?
- What patterns emerge when the market closes or during anomalies?
- Are stock returns stationary? How do autocorrelation patterns differ by sector?
- Which sectors have the highest Value at Risk?
- Can ARIMA/GARCH models effectively capture volatility clustering?
- What are the outliers and what market conditions produce them?

---

## Tools & Technologies

- **Python:** pandas, numpy, scipy, statsmodels, scikit-learn
- **Visualization:** matplotlib, seaborn, plotly
- **Data Sources:** Yahoo Finance API (yfinance), Kaggle dataset
- **Statistical Analysis:** Time series decomposition, ARIMA, GARCH, VaR
- **Dashboards:** Power BI, Plotly, Jupyter interactive widgets

---

## Expected Outputs

1. **EDA Report:** Summary statistics, distributions, and initial insights
2. **Trend Analysis:** Price trends by sector with visualizations
3. **Volatility Study:** GARCH models, volatility clustering identification
4. **Time Series Models:** ARIMA fits with diagnostic plots
5. **Risk Analysis:** VaR estimates, risk-adjusted returns by sector
6. **Interactive Dashboard:** Comprehensive visualization of key metrics
7. **Research Paper:** Findings, methodology, and conclusions

---

## Next Steps

- [ ] Download and load Kaggle dataset
- [ ] Set up Yahoo Finance API integration
- [ ] Perform initial EDA and data quality checks
- [ ] Develop sector comparison framework
- [ ] Implement time series models
- [ ] Create visualizations and dashboards
- [ ] Document findings and conclusions

---

## References & Literature

**Key Topics:**
- Time Series Forecasting and Autocorrelation (Chapters 20-21)
- Autoregressive Models (AR, ARIMA, ARIMAX)
- Volatility Clustering and GARCH Models
- Value at Risk (VaR) and Risk Metrics
- Stationarity Testing (ADF Test)
- Asset Pricing Models and Stock Returns

See `literature/references.md` for detailed citations and reading materials.

---

## Author

**Marianandrade23**  
Research conducted under guidance of **Dr. Sandstrom**

---

## License

This project is for educational and research purposes.

---

## Contact & Support

For questions or discussions about this research, please open an issue in the repository.
