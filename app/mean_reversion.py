import sys
sys.path.append('/home/sierra1/projects/meanReversion')
from config import SQL_ALCHEMY_CONN

import ta 

import pandas as pd 
from sqlalchemy import create_engine
import numpy as np

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
SELECT f.SYMBOL, f.DATE,f.HIGH, f.LOW,  f.CLOSE, f.VOLUME
FROM NSEDATA_FACT f
INNER JOIN METADATA2 m ON f.SYMBOL = m.SYMBOL 
WHERE m.LISTING_DATE <= CURRENT_DATE - INTERVAL '30 days'
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

def calculate_adx(df, window=14):

    df['up_move'] = df.groupby('symbol')['high'].diff()
    df['down_move'] = -df.groupby('symbol')['low'].diff()

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

    #df['ADX'] = df['DX'].ewm(alpha=1/window, adjust=False).mean()
    df['ADX'] = np.nan
    adx_start = df['DX'].rolling(window=window).mean()
    df.loc[df.index[window-1], 'ADX'] = adx_start.iloc[window-1]

    for i in range(window, len(df)):
        prev_adx = df.loc[df.index[i - 1], 'ADX']
        current_dx = df.loc[df.index[i], 'DX']
        df.loc[df.index[i], 'ADX'] = ((prev_adx * (window - 1)) + current_dx) / window

    return df[['date', '+DI', '-DI', 'ADX']]



#print(df.head())
df['date'] = pd.to_datetime(df['date'])


## df has ['SYMBOL', 'DATE', 'CLOSE', 'VOLUME']
df = df.sort_values(by=['symbol', 'date'], ascending=[True, True])  # Very important!

## Calculate 20-day EMA for each stock symbol
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
df['20D_Z_SCORE'] = df.groupby('symbol')['close'].transform(
    lambda x: (x - x.ewm(span=20, adjust=False).mean()) / x.rolling(20).std()
)

df['45D_Z_SCORE'] = df.groupby('symbol')['close'].transform(
    lambda x: (x - x.ewm(span=45, adjust=False).mean()) / x.rolling(45).std()
)

adx_result = (
    df.groupby('symbol')
      .apply(calculate_adx, window=45)
)

df = df.merge(adx_result, on=['symbol', 'date'], how='left')

#Calculating DI difference for fading momentum
# Compute di_diff per stock
# df['di_diff'] = df.groupby('symbol')['+DI'].transform(lambda x: x - df['-DI'])

# # Compute slope (first difference)
# df['di_diff_slope'] = df.groupby('symbol')['di_diff'].transform(lambda x: x.diff())

# # Rolling 25th percentile threshold of di_diff
# df['rolling_quantile_threshold'] = df.groupby('symbol')['di_diff'].transform(
#     lambda x: x.rolling(window=45).quantile(0.25)
# )

# # Fading momentum condition
# df['fading_momentum'] = (
#     (df['di_diff'] < df['rolling_quantile_threshold']) &
#     (df['di_diff_slope'] < 0) &
#     (df['+DI'] > df['-DI'])
# )

# # 70th percentile filter for +DI
# df['low_DI_filter'] = df.groupby('symbol')['+DI'].transform(
#     lambda x: x.rolling(45).quantile(0.70)
# )


# df['Buy_Signal'] = df['45D_Z_SCORE'] < -1.85
# df['Sell_Signal'] = df['45D_Z_SCORE'] > 1.85
# df['Strong_Buy'] = (df['45D_Z_SCORE'] < -2 )& (df['RSI_7'] < 37) | (df['RSI_7'] < 22)
# df['Strong_Sell'] = (
#     df['fading_momentum'] &
#     ((df['45D_Z_SCORE'] > 2) & (df['RSI_7'] > 67)) &
#     df['low_DI_filter']
# )


#Resorting the data, for latest dates to be on top
df = df.sort_values(by=['symbol', 'date'], ascending=[True, False])

# Check the last few rows for 'close' column NaNs or issues
#print(df.tail(30))

df_first_row=df.groupby('symbol').head(1)
print(df_first_row)
df_second_row=df.groupby('symbol').nth(1).reset_index()
#print(df_second_row)


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

