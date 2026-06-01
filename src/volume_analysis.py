"""
Volume Analysis Module
Calculate volume indicators: OBV, VROC, VWAP, MFI, Price-Volume Trend
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VolumeAnalyzer:
    """
    Calculate volume-based indicators
    """
    
    @staticmethod
    def on_balance_volume(df, volume_col='Volume', price_col='Close'):
        """
        Calculate On-Balance Volume (OBV)
        
        Cumulative volume with sign based on price direction:
        - Positive volume if Close > previous Close
        - Negative volume if Close < previous Close
        - No change if Close = previous Close
        
        Args:
            df (pd.DataFrame): OHLCV data (sorted by date)
            volume_col (str): Column name for volume
            price_col (str): Column name for price
        
        Returns:
            pd.Series: OBV values
        """
        obv = [0]
        
        for i in range(1, len(df)):
            if df[price_col].iloc[i] > df[price_col].iloc[i-1]:
                obv.append(obv[-1] + df[volume_col].iloc[i])
            elif df[price_col].iloc[i] < df[price_col].iloc[i-1]:
                obv.append(obv[-1] - df[volume_col].iloc[i])
            else:
                obv.append(obv[-1])
        
        return pd.Series(obv, index=df.index, name='OBV')
    
    @staticmethod
    def volume_rate_of_change(df, volume_col='Volume', period=20):
        """
        Calculate Volume Rate of Change (VROC)
        
        Momentum of trading volume:
        VROC = (Volume_t - Volume_{t-n}) / Volume_{t-n} * 100
        
        Args:
            df (pd.DataFrame): OHLCV data
            volume_col (str): Column name for volume
            period (int): Lookback period (default 20 days)
        
        Returns:
            pd.Series: VROC values (percentage)
        """
        vroc = ((df[volume_col] - df[volume_col].shift(period)) / df[volume_col].shift(period)) * 100
        return vroc.rename('VROC')
    
    @staticmethod
    def volume_weighted_average_price(df, price_col='Close', volume_col='Volume', period=20):
        """
        Calculate Volume Weighted Average Price (VWAP)
        
        VWAP = Σ(Price × Volume) / Σ(Volume)
        
        Args:
            df (pd.DataFrame): OHLCV data
            price_col (str): Column name for price
            volume_col (str): Column name for volume
            period (int): Lookback period for rolling calculation
        
        Returns:
            pd.Series: VWAP values
        """
        tp = df[price_col]  # Typical Price
        vwap = (tp * df[volume_col]).rolling(period).sum() / df[volume_col].rolling(period).sum()
        return vwap.rename('VWAP')
    
    @staticmethod
    def price_volume_trend(df, price_col='Close', volume_col='Volume'):
        """
        Calculate Price-Volume Trend (PVT)
        
        Combines price momentum with volume:
        PVT = PVT_{t-1} + Volume * (Close_t - Close_{t-1}) / Close_{t-1}
        
        Args:
            df (pd.DataFrame): OHLCV data
            price_col (str): Column name for price
            volume_col (str): Column name for volume
        
        Returns:
            pd.Series: PVT values
        """
        pvt = [0]
        returns = df[price_col].pct_change()
        
        for i in range(1, len(df)):
            pvt_change = df[volume_col].iloc[i] * returns.iloc[i]
            pvt.append(pvt[-1] + pvt_change)
        
        return pd.Series(pvt, index=df.index, name='PVT')
    
    @staticmethod
    def money_flow_index(df, period=14, high_col='High', low_col='Low', 
                        close_col='Close', volume_col='Volume'):
        """
        Calculate Money Flow Index (MFI)
        
        RSI-like indicator incorporating volume. Ranges 0-100.
        
        Args:
            df (pd.DataFrame): OHLCV data
            period (int): Lookback period (default 14)
            high_col (str): Column name for high price
            low_col (str): Column name for low price
            close_col (str): Column name for close price
            volume_col (str): Column name for volume
        
        Returns:
            pd.Series: MFI values (0-100)
        """
        # Typical Price
        tp = (df[high_col] + df[low_col] + df[close_col]) / 3
        
        # Raw Money Flow
        rmf = tp * df[volume_col]
        
        # Positive and Negative Money Flow
        positive_mf = [0] * len(df)
        negative_mf = [0] * len(df)
        
        for i in range(1, len(df)):
            if tp.iloc[i] > tp.iloc[i-1]:
                positive_mf[i] = rmf.iloc[i]
            else:
                negative_mf[i] = rmf.iloc[i]
        
        # Money Flow Ratio
        positive_mf_sum = pd.Series(positive_mf).rolling(period).sum()
        negative_mf_sum = pd.Series(negative_mf).rolling(period).sum()
        
        money_flow_ratio = positive_mf_sum / negative_mf_sum.replace(0, 1)
        
        # Money Flow Index
        mfi = 100 - (100 / (1 + money_flow_ratio))
        return mfi.rename('MFI')
    
    @staticmethod
    def volume_moving_average(df, volume_col='Volume', period=20):
        """
        Calculate moving average of volume
        
        Args:
            df (pd.DataFrame): OHLCV data
            volume_col (str): Column name for volume
            period (int): Moving average period
        
        Returns:
            pd.Series: Volume MA
        """
        return df[volume_col].rolling(period).mean().rename(f'VolMA_{period}')
    
    @staticmethod
    def volume_spike_detector(df, volume_col='Volume', std_threshold=2, period=20):
        """
        Detect volume spikes above moving average + N standard deviations
        
        Args:
            df (pd.DataFrame): OHLCV data
            volume_col (str): Column name for volume
            std_threshold (float): Number of std devs above MA to flag spike
            period (int): MA/std lookback period
        
        Returns:
            pd.DataFrame: Data with spike detection columns
        """
        vol_ma = df[volume_col].rolling(period).mean()
        vol_std = df[volume_col].rolling(period).std()
        
        upper_band = vol_ma + (std_threshold * vol_std)
        
        df['VolumeSpike'] = (df[volume_col] > upper_band).astype(int)
        df['VolumeSpikeRatio'] = df[volume_col] / vol_ma
        
        return df[['VolumeSpike', 'VolumeSpikeRatio']]


class VolumeStatistics:
    """
    Calculate volume statistics for analysis
    """
    
    @staticmethod
    def volume_summary_stats(df, volume_col='Volume', groupby_col=None):
        """
        Calculate volume summary statistics
        
        Args:
            df (pd.DataFrame): OHLCV data
            volume_col (str): Column name for volume
            groupby_col (str): Optional column to group by (e.g., 'Sector')
        
        Returns:
            pd.DataFrame: Summary statistics
        """
        if groupby_col:
            stats = df.groupby(groupby_col)[volume_col].agg([
                'count', 'mean', 'std', 'min', 'max', 'median'
            ]).round(2)
        else:
            stats = pd.DataFrame({
                'count': [df[volume_col].count()],
                'mean': [df[volume_col].mean()],
                'std': [df[volume_col].std()],
                'min': [df[volume_col].min()],
                'max': [df[volume_col].max()],
                'median': [df[volume_col].median()]
            }).round(2)
        
        return stats
    
    @staticmethod
    def average_daily_volume(df, volume_col='Volume', groupby_col=None):
        """
        Calculate average daily volume (ADV)
        
        Args:
            df (pd.DataFrame): OHLCV data
            volume_col (str): Column name for volume
            groupby_col (str): Optional grouping column
        
        Returns:
            float or pd.Series: Average daily volume
        """
        if groupby_col:
            adv = df.groupby(groupby_col)[volume_col].mean()
            return adv.round(0)
        else:
            return df[volume_col].mean()


def calculate_all_volume_indicators(df):
    """
    Calculate all volume indicators in one function
    
    Args:
        df (pd.DataFrame): OHLCV data (must have Date, Close, Volume, High, Low)
    
    Returns:
        pd.DataFrame: Original data plus all volume indicators
    """
    analyzer = VolumeAnalyzer()
    
    df['OBV'] = analyzer.on_balance_volume(df)
    df['VROC'] = analyzer.volume_rate_of_change(df)
    df['VWAP'] = analyzer.volume_weighted_average_price(df)
    df['PVT'] = analyzer.price_volume_trend(df)
    df['MFI'] = analyzer.money_flow_index(df)
    df['VolMA_20'] = analyzer.volume_moving_average(df)
    
    spikes = analyzer.volume_spike_detector(df)
    df['VolumeSpike'] = spikes['VolumeSpike']
    df['VolumeSpikeRatio'] = spikes['VolumeSpikeRatio']
    
    return df


# Example usage
if __name__ == "__main__":
    print("Volume analysis module ready for use")
