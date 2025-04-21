import sys
sys.path.append('/home/sierra1/projects/meanReversion')
from config import SQL_ALCHEMY_CONN

from ta.trend import ADXIndicator

import pandas as pd 
from sqlalchemy import create_engine
import numpy as np

from datetime import timedelta


engine = create_engine(SQL_ALCHEMY_CONN)
buy_reco = set()
sell_reco = set()
strong_buy_reco = set()
strong_sell_reco = set()

buy_reco_ystd = set()
sell_reco_ystd = set()
strong_buy_reco_ystd = set()
strong_sell_reco_ystd = set()

query="""
SELECT f.SYMBOL, f.DATE,f.OPEN, f.HIGH, f.LOW,  f.CLOSE, f.VOLUME
FROM NSEDATA_FACT f
INNER JOIN METADATA2 m ON f.SYMBOL = m.SYMBOL 
WHERE m.LISTING_DATE <= CURRENT_DATE - INTERVAL '65 days'
and DATE >= CURRENT_DATE - INTERVAL '240 days';
"""

df = pd.read_sql(query, engine)

def calculate_rsi(close: pd.Series, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calculate_adx(df, window=45): #Propritory method
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

    return df[['date','symbol', '+DI', '-DI', 'ADX']]


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

#setting date time format
df['date'] = pd.to_datetime(df['date'])



df = df.sort_values(by=['symbol', 'date'], ascending=[True, True])  # Very important!



#Calculating EMA, RSI and RSI-EMA
df['EMA_10'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=10, adjust=False).mean())
df['EMA_20'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
df['EMA_45'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=45, adjust=False).mean())
df['EMA_60'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=60, adjust=False).mean())
df['EMA_90'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=90, adjust=False).mean())
df['EMA_120'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=120, adjust=False).mean())
df['RSI_14'] = df.groupby('symbol')['close'].transform(lambda x: calculate_rsi(x, window=14))
df['RSI_7']  = df.groupby('symbol')['close'].transform(lambda x: calculate_rsi(x, window=7))
df['RSI_EMA_14'] = df.groupby('symbol')['RSI_7'].transform(lambda x: x.ewm(span=14, adjust=False).mean())



# Calculate z-scores directly


df['45D_Z_SCORE_CLOSE'] = df.groupby('symbol')['close'].transform(
    lambda x: (x - x.ewm(span=45, adjust=False).mean()) / x.rolling(45).std()
)



df['45D_Z_SCORE_OPEN'] = df.groupby('symbol')['open'].transform(
    lambda x: (x - x.ewm(span=45, adjust=False).mean()) / x.rolling(45).std()
) 

# We will be using open/close for 45D score delibrately for better confirmation 


# Group by symbol first, then apply the function on the dataframe excluding the symbol
# adx_ta = df.groupby('symbol', group_keys=False).apply(lambda g: calculate_adx_ta(g, window=45))
# df = df.merge(adx_ta, on=['symbol', 'date'], how='left') #Using ADXIndicator from ta library
adx_prop = df.groupby('symbol', group_keys=False).apply(lambda g: calculate_adx(g, window=15)) #Using proprietory function 
df = df.merge(adx_prop, on=['symbol', 'date'], how='left')


#Calculating DI difference for fading momentum
# Compute di_diff per stock
df['di_diff'] = df['+DI'] - df['-DI']
df['di_diff_20D_zscore'] = df.groupby('symbol')['di_diff'].transform(lambda x: (x - x.rolling(window=20).mean()) / x.rolling(20).std())

# df['rolling_quantile_threshold'] = df.groupby('symbol')['di_diff'].transform(
#     lambda x: x.rolling(window=45).quantile(0.7)
# )

# df['fading_momentum'] = (
#     (df['di_diff'] > df['rolling_quantile_threshold']) &
#     (df['di_diff_slope'] < 0) &
#     (df['+DI'] > df['-DI'])
# )

# df['DI_45D_70perc_filter'] = df.groupby('symbol')['+DI'].transform(
#     lambda x: x.rolling(45).quantile(0.70)
# )
# (
#     df['fading_momentum'] &
#     ((df['45D_Z_SCORE'] > 2) & (df['RSI_7'] > 67)) &
#     ((df['+DI']) > (df['DI_45D_70perc_filter'])) |
    

# )
#df['Buy_Signal'] = df['45D_Z_SCORE_CLOSE'] <= -1.85
#df['Sell_Signal'] = df['45D_Z_SCORE_OPEN'] >= 1.85
df['20D_Z_SCORE_OPEN'] = df.groupby('symbol')['open'].transform(
    lambda x: (x - x.ewm(span=20, adjust=False).mean()) / x.rolling(20).std()
)
df['20D_Z_SCORE_CLOSE'] = df.groupby('symbol')['close'].transform(
    lambda x: (x - x.rolling(window=20).mean()) / x.rolling(20).std()
)

df['Strong_Buy'] = ((df['20D_Z_SCORE_CLOSE'] <= -2.1) | ((df['di_diff_20D_zscore'] <= -2) & (df['20D_Z_SCORE_CLOSE'] <= -2)))
df['Strong_Sell'] = (df['20D_Z_SCORE_OPEN'] >= 2.2) | ((df['di_diff_20D_zscore'] >= 2) & (df['20D_Z_SCORE_OPEN'] >=2))


latest_date = df['date'].max()
cutoff_date = latest_date - timedelta(days=10)

recent_data= df[df['date'] >= cutoff_date]

strong_buy_symbols = recent_data[recent_data['Strong_Buy'] == 1]['symbol'].unique()
strong_sell_symbols = recent_data[recent_data['Strong_Sell'] == 1]['symbol'].unique()
# df['Stop_Loss'] = 



#Resorting the data, for latest dates to be on top
df = df.sort_values(by=['symbol', 'date'], ascending=[True, False])
print(df[df["symbol"] == "AFIL"].head(15))
df_first_row=df.groupby('symbol').head(1)
# print(df_first_row)
# print(f'BUY - {strong_buy_symbols}')
# print(f'SELL - {strong_sell_symbols}')


# buy_reco.update(df_first_row[df_first_row['Buy_Signal'] == True]['symbol'])
# sell_reco.update(df_first_row[df_first_row['Sell_Signal'] == True]['symbol'])
# strong_buy_reco.update(df_first_row[df_first_row['Strong_Buy'] == True]['symbol'])
# strong_sell_reco.update(df_first_row[df_first_row['Strong_Sell'] == True]['symbol'])

# buy_reco_ystd.update(df_second_row[df_second_row['Buy_Signal'] == True]['symbol'])
# sell_reco_ystd.update(df_second_row[df_second_row['Sell_Signal'] == True]['symbol'])
# strong_buy_reco_ystd.update(df_second_row[df_second_row['Strong_Buy'] == True]['symbol'])
# strong_sell_reco_ystd.update(df_second_row[df_second_row['Strong_Sell'] == True]['symbol'])

# print(f'buy recommendation - {buy_reco}')
# print(f'sell recommendation - {sell_reco}')
# print(f'Stong buy recommendation - {strong_buy_reco}')
# print(f'Stong sell recommendation - {strong_sell_reco}')

# print(f'buy recommendation for ystd - {buy_reco_ystd}')
# print(f'sell recommendation for ystd - {sell_reco_ystd}')
# print(f'Stong buy recommendation for ystd - {strong_buy_reco_ystd}')
# print(f'Stong sell recommendation for ystd - {strong_sell_reco_ystd}')

