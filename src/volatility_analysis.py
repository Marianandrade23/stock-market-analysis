"""
Volatility Analysis Module
GARCH modeling, rolling volatility, volatility clustering detection
"""

import pandas as pd
import numpy as np
import logging
from arch import arch_model
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VolatilityCalculator:
    """
    Calculate volatility metrics
    """
    
    @staticmethod
    def log_returns(df, price_col='Close'):
        """
        Calculate log returns
        
        log_return = ln(Price_t / Price_{t-1})
        
        Args:
            df (pd.DataFrame): OHLCV data
            price_col (str): Price column name
        
        Returns:
            pd.Series: Log returns
        """
        returns = np.log(df[price_col] / df[price_col].shift(1))
        return returns.rename('LogReturns')
    
    @staticmethod
    def simple_returns(df, price_col='Close'):
        """
        Calculate simple returns (percentage change)
        
        simple_return = (Price_t - Price_{t-1}) / Price_{t-1}
        
        Args:
            df (pd.DataFrame): OHLCV data
            price_col (str): Price column name
        
        Returns:
            pd.Series: Simple returns
        """
        returns = df[price_col].pct_change()
        return returns.rename('SimpleReturns')
    
    @staticmethod
    def rolling_volatility(df, returns_col='LogReturns', period=20, annualize=True):
        """
        Calculate rolling volatility (standard deviation of returns)
        
        Args:
            df (pd.DataFrame): Data with returns column
            returns_col (str): Column name with returns
            period (int): Window size (default 20 trading days ≈ 1 month)
            annualize (bool): Annualize volatility (multiply by sqrt(252))
        
        Returns:
            pd.Series: Rolling volatility
        """
        rolling_std = df[returns_col].rolling(period).std()
        
        if annualize:
            rolling_std = rolling_std * np.sqrt(252)  # 252 trading days per year
        
        return rolling_std.rename(f'RollingVol_{period}')
    
    @staticmethod
    def exponential_weighted_volatility(df, returns_col='LogReturns', span=20, annualize=True):
        """
        Calculate exponentially weighted volatility (newer data weighted more heavily)
        
        Args:
            df (pd.DataFrame): Data with returns column
            returns_col (str): Column name with returns
            span (int): Span for exponential weighting
            annualize (bool): Annualize volatility
        
        Returns:
            pd.Series: Exponentially weighted volatility
        """
        ewm_std = df[returns_col].ewm(span=span).std()
        
        if annualize:
            ewm_std = ewm_std * np.sqrt(252)
        
        return ewm_std.rename(f'EWMVol_{span}')
    
    @staticmethod
    def parkinson_volatility(df, high_col='High', low_col='Low', period=20, annualize=True):
        """
        Calculate Parkinson volatility using High-Low range
        More efficient than close-to-close volatility
        
        Args:
            df (pd.DataFrame): OHLCV data
            high_col (str): High price column
            low_col (str): Low price column
            period (int): Window size
            annualize (bool): Annualize volatility
        
        Returns:
            pd.Series: Parkinson volatility
        """
        hl_ratio = np.log(df[high_col] / df[low_col])
        parkinson_vol = (1 / (4 * np.log(2))) * (hl_ratio ** 2)
        rolling_vol = parkinson_vol.rolling(period).mean()
        
        if annualize:
            rolling_vol = np.sqrt(rolling_vol) * np.sqrt(252)
        
        return rolling_vol.rename(f'ParkinsonVol_{period}')


class GARCHModel:
    """
    GARCH(1,1) modeling for volatility
    """
    
    @staticmethod
    def fit_garch(returns, p=1, q=1):
        """
        Fit GARCH(p,q) model
        
        Variance equation: σ_t² = ω + Σ α_i*ε_{t-i}² + Σ β_j*σ_{t-j}²
        
        Args:
            returns (pd.Series): Returns series (as percentage, e.g., * 100)
            p (int): ARCH order (default 1)
            q (int): GARCH order (default 1)
        
        Returns:
            ArchModelResult: Fitted GARCH model
        """
        try:
            model = arch_model(returns, vol='Garch', p=p, q=q)
            results = model.fit(disp='off')
            logger.info(f"GARCH({p},{q}) fitted successfully")
            return results
        except Exception as e:
            logger.error(f"Error fitting GARCH model: {e}")
            return None
    
    @staticmethod
    def extract_garch_parameters(garch_results):
        """
        Extract key GARCH parameters
        
        Args:
            garch_results: Fitted GARCH model results
        
        Returns:
            dict: Parameters {omega, alpha, beta, persistence}
        """
        params = {
            'omega': garch_results.params['omega'],
            'alpha': garch_results.params.get('alpha[1]', np.nan),
            'beta': garch_results.params.get('beta[1]', np.nan),
        }
        
        params['persistence'] = params['alpha'] + params['beta']
        params['mean'] = garch_results.params.get('mu', np.nan)
        
        return params
    
    @staticmethod
    def garch_conditional_volatility(garch_results, annualize=True):
        """
        Extract conditional volatility from fitted GARCH
        
        Args:
            garch_results: Fitted GARCH model results
            annualize (bool): Annualize volatility
        
        Returns:
            pd.Series: Conditional volatility
        """
        cond_vol = garch_results.conditional_volatility
        
        if annualize:
            cond_vol = cond_vol * np.sqrt(252)
        
        return cond_vol
    
    @staticmethod
    def forecast_garch_volatility(garch_results, horizon=20, annualize=True):
        """
        Forecast volatility using GARCH model
        
        Args:
            garch_results: Fitted GARCH model results
            horizon (int): Number of steps ahead to forecast
            annualize (bool): Annualize volatility
        
        Returns:
            pd.DataFrame: Forecasted volatility
        """
        forecast = garch_results.forecast(horizon=horizon)
        conditional_vol = np.sqrt(forecast.variance.values[-1, :])
        
        if annualize:
            conditional_vol = conditional_vol * np.sqrt(252)
        
        return pd.DataFrame({
            'Horizon': range(1, horizon + 1),
            'ForecastedVolatility': conditional_vol
        })
    
    @staticmethod
    def garch_diagnostics(garch_results):
        """
        Calculate GARCH model diagnostics
        
        Args:
            garch_results: Fitted GARCH model results
        
        Returns:
            dict: Diagnostic metrics
        """
        residuals = garch_results.resid
        standardized_residuals = residuals / garch_results.conditional_volatility
        
        diagnostics = {
            'AIC': garch_results.aic,
            'BIC': garch_results.bic,
            'Log-Likelihood': garch_results.loglikelihood,
            'Mean_Residuals': residuals.mean(),
            'Std_Residuals': residuals.std(),
            'Mean_Std_Residuals': standardized_residuals.mean(),
            'Std_Std_Residuals': standardized_residuals.std(),
            'Ljung_Box_Q_Stat': garch_results.resid.autocorr()  # Simplified
        }
        
        return diagnostics


class VolatilityAnalysis:
    """
    Advanced volatility analysis and regime detection
    """
    
    @staticmethod
    def volatility_clustering(returns, window=20, threshold_percentile=75):
        """
        Detect volatility clustering (high vol periods)
        
        Args:
            returns (pd.Series): Returns series
            window (int): Rolling window for volatility
            threshold_percentile (int): Percentile to define high volatility
        
        Returns:
            pd.DataFrame: Clustering detection
        """
        rolling_vol = returns.rolling(window).std()
        threshold = rolling_vol.quantile(threshold_percentile / 100)
        
        clustering = pd.DataFrame({
            'Rolling_Volatility': rolling_vol,
            'IsHighVol': (rolling_vol > threshold).astype(int),
            'VolThreshold': threshold
        })
        
        return clustering
    
    @staticmethod
    def volatility_regimes(returns, num_regimes=3, window=20):
        """
        Identify volatility regimes (low, normal, high)
        
        Args:
            returns (pd.Series): Returns series
            num_regimes (int): Number of regimes to identify (default 3)
            window (int): Rolling window for volatility
        
        Returns:
            pd.DataFrame: Regime classification
        """
        rolling_vol = returns.rolling(window).std()
        
        # Use quantile-based regime classification
        quantiles = rolling_vol.quantile([1/num_regimes, 2/num_regimes])
        
        regimes = pd.cut(rolling_vol, bins=[0] + list(quantiles) + [rolling_vol.max()],
                         labels=[f'Regime_{i}' for i in range(num_regimes)],
                         ordered=True)
        
        return pd.DataFrame({
            'Rolling_Volatility': rolling_vol,
            'Regime': regimes
        })
    
    @staticmethod
    def realized_volatility(df, price_col='Close', period=20, annualize=True):
        """
        Calculate realized volatility from intraday returns
        
        Args:
            df (pd.DataFrame): OHLCV data
            price_col (str): Price column
            period (int): Window for realized vol
            annualize (bool): Annualize
        
        Returns:
            pd.Series: Realized volatility
        """
        returns = np.log(df[price_col] / df[price_col].shift(1))
        realized_vol = returns.rolling(period).std()
        
        if annualize:
            realized_vol = realized_vol * np.sqrt(252)
        
        return realized_vol
    
    @staticmethod
    def volatility_jumps(returns, threshold_std=3):
        """
        Detect volatility jumps (sudden large moves)
        
        Args:
            returns (pd.Series): Returns series
            threshold_std (float): Threshold in standard deviations
        
        Returns:
            pd.DataFrame: Jump detection
        """
        mean_return = returns.mean()
        std_return = returns.std()
        
        jumps = pd.DataFrame({
            'Returns': returns,
            'Z_Score': (returns - mean_return) / std_return,
            'IsJump': (np.abs(returns - mean_return) > threshold_std * std_return).astype(int)
        })
        
        return jumps


def fit_garch_by_sector(df, groupby_col='Sector'):
    """
    Fit GARCH models for each sector
    
    Args:
        df (pd.DataFrame): Data with returns and sector column
        groupby_col (str): Column to group by (default 'Sector')
    
    Returns:
        dict: GARCH results and parameters by sector
    """
    garch_results = {}
    
    for sector in df[groupby_col].unique():
        sector_data = df[df[groupby_col] == sector]['LogReturns'].dropna() * 100
        
        if len(sector_data) > 50:  # Need sufficient data
            model = arch_model(sector_data, vol='Garch', p=1, q=1)
            try:
                results = model.fit(disp='off')
                params = GARCHModel.extract_garch_parameters(results)
                garch_results[sector] = {
                    'model': results,
                    'parameters': params
                }
                logger.info(f"GARCH fitted for {sector}: persistence = {params['persistence']:.3f}")
            except Exception as e:
                logger.warning(f"Failed to fit GARCH for {sector}: {e}")
    
    return garch_results


# Example usage
if __name__ == "__main__":
    print("Volatility analysis module ready for use")
