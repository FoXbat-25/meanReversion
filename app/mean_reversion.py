import sys
sys.path.append('/home/sierra1/projects/meanReversion')
from config import SQL_ALCHEMY_CONN

import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(SQL_ALCHEMY_CONN)
buy_reco = set()
sell_reco = set()
strong_buy_reco = set()
strong_sell_reco = set()

query="""
SELECT f.SYMBOL, f.DATE, f.CLOSE, f.VOLUME
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


#print(df.head())
df['date'] = pd.to_datetime(df['date'])


## df has ['SYMBOL', 'DATE', 'CLOSE', 'VOLUME']
df = df.sort_values(by=['symbol', 'date'], ascending=[True, True])  # Very important!

## Calculate 20-day EMA for each stock symbol
df['EMA_20'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
df['EMA_45'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=45, adjust=False).mean())
df['EMA_60'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=60, adjust=False).mean())
df['EMA_90'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=90, adjust=False).mean())
df['EMA_120'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=120, adjust=False).mean())
df['RSI_14'] = df.groupby('symbol')['close'].transform(lambda x: calculate_rsi(x, window=14))
df['RSI_7']  = df.groupby('symbol')['close'].transform(lambda x: calculate_rsi(x, window=7))
df['RSI_EMA_14'] = df.groupby('symbol')['RSI_7'].transform(lambda x: x.ewm(span=14, adjust=False).mean())

# Calculate standard deviation for each group
# df['rolling_std_20'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(20).std())
# df['rolling_std_45'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(45).std())

# Then calculate Z-scores directly without using apply
# df['20D_Z_SCORE'] = (df['close'] - df['EMA_20']) / df['rolling_std_20']
# df['45D_Z_SCORE'] = (df['close'] - df['EMA_45']) / df['rolling_std_45']

# Calculate z-scores directly
df['20D_Z_SCORE'] = df.groupby('symbol')['close'].transform(
    lambda x: (x - x.ewm(span=20, adjust=False).mean()) / x.rolling(20).std()
)

df['45D_Z_SCORE'] = df.groupby('symbol')['close'].transform(
    lambda x: (x - x.ewm(span=45, adjust=False).mean()) / x.rolling(45).std()
)

#Resorting the data, for latest dates to be on top
df = df.sort_values(by=['symbol', 'date'], ascending=[True, False])

df['Buy_Signal'] = df['20D_Z_SCORE'] < -1.5
df['Sell_Signal'] = df['20D_Z_SCORE'] > 1.5
df['Strong_Buy'] = (df['20D_Z_SCORE'] < -1.85 )& (df['RSI_7'] < 37)
df['Strong_Sell'] = (df['20D_Z_SCORE'] > 1.85) & (df['RSI_7'] > 67)

# Check the last few rows for 'close' column NaNs or issues
#print(df.tail(30))

df_first_row=df.groupby('symbol').head(1)
#print(df_first_row)

buy_reco.update(df_first_row[df_first_row['Buy_Signal'] == True]['symbol'])
sell_reco.update(df_first_row[df_first_row['Sell_Signal'] == True]['symbol'])
strong_buy_reco.update(df_first_row[df_first_row['Strong_Buy'] == True]['symbol'])
strong_sell_reco.update(df_first_row[df_first_row['Strong_Sell'] == True]['symbol'])

print(f'buy recommendation - {buy_reco}')
print(f'sell recommendation - {sell_reco}')
print(f'Stong buy recommendation - {strong_buy_reco}')
print(f'Stong sell recommendation - {strong_sell_reco}')
