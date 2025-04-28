import sys
sys.path.append('/home/sierra1/projects/meanReversion')
from config import SQL_ALCHEMY_CONN

from ta.trend import ADXIndicator

import pandas as pd 
from sqlalchemy import create_engine
import numpy as np

buy_reco = set()
sell_reco = set()
strong_buy_reco = set()
strong_sell_reco = set()

buy_reco_ystd = set()
sell_reco_ystd = set()
strong_buy_reco_ystd = set()
strong_sell_reco_ystd = set()

df = pd.read_sql() # (query, engine)
df['45D_Z_SCORE_CLOSE'] = df.groupby('symbol')['close'].transform(
    lambda x: (x - x.ewm(span=45, adjust=False).mean()) / x.rolling(45).std()
)
df['45D_Z_SCORE_OPEN'] = df.groupby('symbol')['open'].transform(
    lambda x: (x - x.ewm(span=45, adjust=False).mean()) / x.rolling(45).std()
) 
df['20D_Z_SCORE_OPEN'] = df.groupby('symbol')['open'].transform(
    lambda x: (x - x.ewm(span=20, adjust=False).mean()) / x.rolling(20).std()
)
df['20D_Z_SCORE_CLOSE'] = df.groupby('symbol')['close'].transform(
    lambda x: (x - x.rolling(window=20).mean()) / x.rolling(20).std()
)

def calculate_adx_ta(df, window=14): # Using ADXIndicator from ta library
    df = df.copy()
    df = df.sort_values('date')
    if df[['high', 'low', 'close']].isnull().any().any():
        return pd.DataFrame()  # Skip broken data

    # Step 2: skip small groups
    if len(df) < window + 1:
        return pd.DataFrame()

    # Step 3: calculate indicators
    try:
        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=window)
        df['+DI'] = adx.adx_pos()
        df['-DI'] = adx.adx_neg()
        df['ADX'] = adx.adx()
        df['symbol'] = df['symbol'].iloc[0]
        return df[['date', 'symbol', '+DI', '-DI', 'ADX']]
    except Exception as e:
        print(f"Failed for a symbol:| Reason: {e}")
        return pd.DataFrame()
    
adx_ta = df.groupby('symbol', group_keys=False).apply(lambda g: calculate_adx_ta(g, window=45))
df = df.merge(adx_ta, on=['symbol', 'date'], how='left') #Using ADXIndicator from ta library