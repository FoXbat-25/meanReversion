import sys
sys.path.append('/home/sierra1/projects/meanReversion/')
from app.config import SQL_ALCHEMY_CONN

import psycopg2

import pandas as pd 
from sqlalchemy import create_engine
import numpy as np

from datetime import datetime, timedelta

#Bollinger Band - 2 Std, 20d SMA
#ADX - 15d
#Di_diff - 2 std, 20d SMA
#ATR - 14d sma

engine = create_engine(SQL_ALCHEMY_CONN)


class utils:

    def __init__(self, start_date=None, end_date = None):
        self.start_date = start_date if start_date else datetime.today().date() - timedelta(days=90)
        self.end_date = end_date if end_date else datetime.today().date()
        self.fetch_nse_data()
        

    def fetch_nse_data(self):
    
        query = """
            SELECT f.SYMBOL, f.DATE, f.OPEN, f.HIGH, f.LOW, f.CLOSE, f.VOLUME
            FROM NSEDATA_FACT f
            INNER JOIN METADATA m ON f.SYMBOL = m.SYMBOL 
            WHERE m.LISTING_DATE <= CURRENT_DATE - INTERVAL '65 days'
            AND f.DATE BETWEEN %(start_date)s AND %(end_date)s;
        """

        self.dataframe = pd.read_sql(query, engine, params={"start_date": self.start_date, "end_date": self.end_date})

        self.dataframe['date'] = pd.to_datetime(self.dataframe['date'])
        self.dataframe = self.dataframe.sort_values(by=['symbol', 'date'], ascending=[True, True])

        self.dataframe['next_open'] = self.dataframe.groupby('symbol')['open'].shift(-1)
        self.dataframe['next_date'] = self.dataframe.groupby('symbol')['date'].shift(-1)
        self.dataframe['prev_date'] = self.dataframe.groupby('symbol')['date'].shift(1)

        return self.dataframe

    def fetch_cooldown_end_date(self, df):

        query="""
            SELECT date, cooldown_end_date
            FROM CALENDAR c
            WHERE holiday = False
            ORDER BY date ASC;
        """

        calendar_df = pd.read_sql(query, engine)
        calendar_df['date'] = pd.to_datetime(calendar_df['date'])

        calendar_df = calendar_df.rename(columns={'date': 'next_date'})

        df['date'] = pd.to_datetime(df['date'])
        df = df.merge(calendar_df, on='next_date', how='left')

        return df

    def rsi_formula(self, close: pd.Series, window=14):

        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def adx_formula(self, df, window=15): #Propritory method
        df = df.copy()
        if df[['high', 'low', 'close']].isnull().any().any():
            return pd.DataFrame()  # Skip broken data

        # Step 2: skip small groups
        df['up_move'] = df['high'].diff()
        df['down_move'] = -df['low'].diff()

        df['+DM'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0.0)
        df['-DM'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0.0)

        df['TR'] = np.maximum.reduce([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ])
        
        df['ATR'] = df['TR'].rolling(window=14, min_periods=1).mean()
        # Wilder’s smoothing with EWM
        df['TR_smoothed'] = df['TR'].ewm(alpha=1/window, adjust=False).mean()
        df['+DM_smoothed'] = df['+DM'].ewm(alpha=1/window, adjust=False).mean()
        df['-DM_smoothed'] = df['-DM'].ewm(alpha=1/window, adjust=False).mean()

        df['+DI'] = 100 * df['+DM_smoothed'] / df['TR_smoothed']
        df['-DI'] = 100 * df['-DM_smoothed'] / df['TR_smoothed']

        df['DI_sum'] = df['+DI'] + df['-DI']
        df['DX'] = np.where(df['DI_sum'] != 0,
                            100 * abs(df['+DI'] - df['-DI']) / df['DI_sum'],
                            0)

        df['ADX'] = df['DX'].ewm(alpha=1/window, adjust=False).mean()

        return df[['date','symbol', '+DI', '-DI', 'ADX', 'ATR']]

    def calc_adx(self, df, window = 15):

        adx_prop = df.groupby('symbol', group_keys=False).apply(lambda g: self.adx_formula(g, window=window)) #Using proprietory function 
        df = df.merge(adx_prop, on=['symbol', 'date'], how='left')
    
        return df

    def calc_ema(self, df, span=[10, 20, 45, 60, 90, 120]):
        for i in span:
            df[f'EMA_{i}'] = df.groupby('symbol')['close'].transform(
                lambda x: x.ewm(span=i, adjust=False).mean()
            )    
        return df

    def calc_rsi(self, df, span = [7,14]): 
        for i in span:
            df[f'RSI_{i}'] = df.groupby('symbol')['close'].transform(
                lambda x: self.rsi_formula(x, window=i)
            )
        return df

    def calc_z_score(self, df, span=[45, 20]): #uses SMA
        for i in span:
            for j in ['open', 'close']:
                df[f'{i}d_z_score_{j}'] = df.groupby('symbol')[j].transform(
                    lambda x: (x - (x.rolling(i).mean())) / (x.rolling(i).std())
                )
        return df

    def calc_di_diff(self, df, rolling_window=20):
        df['di_diff'] = df['+DI'] - df['-DI']
        df[f'di_diff_{rolling_window}D_zscore'] = df.groupby('symbol')['di_diff'].transform(
            lambda x: (x - x.rolling(window=rolling_window).mean()) / x.rolling(rolling_window).std()
        )
        return df
    
    def volume_check(self, df, min_volume=500000, rolling_window=20):
        df[f'volume_{rolling_window}d_SMA'] = df['volume'].rolling(window=rolling_window).mean()
        df['volume_flag'] = df[f'volume_{rolling_window}d_SMA'] > min_volume # or ₹10 crore for traded value
        return df
        