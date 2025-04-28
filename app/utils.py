import sys
sys.path.append('/home/sierra1/projects/meanReversion')
from config import SQL_ALCHEMY_CONN, SQL_POSTGRES_CONN

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
        self.start_date = start_date if start_date else datetime.today().date() - timedelta(days=240)
        self.end_date = end_date if end_date else datetime.today().date()
        self.fetch_nse_data()
        

    def fetch_nse_data(self):
    
        query = """
            SELECT f.SYMBOL, f.DATE, f.OPEN, f.HIGH, f.LOW, f.CLOSE, f.VOLUME
            FROM NSEDATA_FACT f
            INNER JOIN METADATA2 m ON f.SYMBOL = m.SYMBOL 
            WHERE m.LISTING_DATE <= CURRENT_DATE - INTERVAL '65 days'
            AND f.DATE BETWEEN %(start_date)s AND %(end_date)s;
        """

        self.dataframe = pd.read_sql(query, engine, params={"start_date": self.start_date, "end_date": self.end_date})
        return self.dataframe


    def rsi_formula(self, close: pd.Series, window=14):

        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calc_adx(self, df, window=15): #Propritory method
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


    def calc_ema(self, df):
        for span in [10, 20, 45, 60, 90, 120]:
            df[f'EMA_{span}'] = df.groupby('symbol')['close'].transform(
                lambda x: x.ewm(span=span, adjust=False).mean()
            )    
        return df

    def calc_rsi(self, df): 
        for span in [7, 14]:
            df[f'RSI_{span}'] = df.groupby('symbol')['close'].transform(
                lambda x: self.rsi_formula(x, window=span)
            )
        return df

    def calc_z_score(self, df, span=[45, 20]): #uses SMA
        for i in span:
            for j in ['open', 'close']:
                df[f'{i}d_z_score_{j}'] = df.groupby('symbol')[j].transform(
                    lambda x: (x - (x.rolling(i).mean())) / (x.rolling(i).std())
                )
        return df

    def calc_di_diff(self, df):
        df['di_diff'] = df['+DI'] - df['-DI']
        df['di_diff_20D_zscore'] = df.groupby('symbol')['di_diff'].transform(
            lambda x: (x - x.rolling(window=20).mean()) / x.rolling(20).std()
        )
        return df

    def next_date(self, df):
        df['next_open'] = df.groupby('symbol')['open'].shift(-1)
        df['next_date'] = df.groupby('symbol')['date'].shift(-1)    

        return df
    


        