import yfinance as yf
import pandas as pd

def load_kaggle_data(path='src/dataset.zip'):
    """Load the Kaggle OHLCV dataset"""
    df = pd.read_csv(path, compression='zip')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Loaded {len(df)} records")
    return df

def fetch_yahoo_data(ticker, start='2020-01-01', end='2026-01-01'):
    """Fetch stock data from Yahoo Finance"""
    df = yf.download(ticker, start=start, end=end)
    return df

def fetch_multiple_tickers(tickers, start='2020-01-01', end='2026-01-01'):
    """Fetch multiple stocks at once"""
    data = yf.download(tickers, start=start, end=end, group_by='ticker')
    return data

# Example usage
if __name__ == "__main__":
    # Test with Apple stock
    aapl = fetch_yahoo_data("AAPL")
    print(aapl.tail())
