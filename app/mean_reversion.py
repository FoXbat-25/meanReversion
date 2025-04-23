import sys
sys.path.append('/home/sierra1/projects/meanReversion')
from config import SQL_ALCHEMY_CONN

from ta.trend import ADXIndicator

import pandas as pd 
from sqlalchemy import create_engine
import numpy as np

from datetime import datetime, timedelta

#Bollinger Band - 2 Std, 20d SMA
#ADX - 15d
#Di_diff - 


engine = create_engine(SQL_ALCHEMY_CONN)
buy_reco = set()
sell_reco = set()
strong_buy_reco = set()
strong_sell_reco = set()

buy_reco_ystd = set()
sell_reco_ystd = set()
strong_buy_reco_ystd = set()
strong_sell_reco_ystd = set()

def fetch_nse_data(start_date=None, end_date=None):
    if start_date is None:
        start_date = datetime.today().date() - timedelta(days=240)
    if end_date is None:
        end_date = datetime.today().date()

    query = """
        SELECT f.SYMBOL, f.DATE, f.OPEN, f.HIGH, f.LOW, f.CLOSE, f.VOLUME
        FROM NSEDATA_FACT f
        INNER JOIN METADATA2 m ON f.SYMBOL = m.SYMBOL 
        WHERE m.LISTING_DATE <= CURRENT_DATE - INTERVAL '65 days'
        AND f.DATE BETWEEN %(start_date)s AND %(end_date)s;
    """

    dataframe = pd.read_sql(query, engine, params={"start_date": start_date, "end_date": end_date})
    return dataframe


def rsi_formula(close: pd.Series, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calc_adx(df, window=15): #Propritory method
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


def calc_ema(df):
    for span in [10, 20, 45, 60, 90, 120]:
        df[f'EMA_{span}'] = df.groupby('symbol')['close'].transform(
            lambda x: x.ewm(span=span, adjust=False).mean()
        )    
    return df

def calc_rsi(df): 
    for span in [7, 14]:
        df[f'RSI_{span}'] = df.groupby('symbol')['close'].transform(
            lambda x: rsi_formula(x, window=span)
        )
    return df

def calc_z_score(df, span=[20, 45]): #uses SMA
    for i in span:
        for j in ['open', 'close']:
            df[f'{i}d_z_score_{j}'] = df.groupby('symbol')[j].transform(
                lambda x: (x - x.rolling(i).mean())/x.rolling(i).std()
            )
    return df

def calc_di_diff(df):
    df['di_diff'] = df['+DI'] - df['-DI']
    df['di_diff_20D_zscore'] = df.groupby('symbol')['di_diff'].transform(
        lambda x: (x - x.rolling(window=20).mean()) / x.rolling(20).std()
    )
    return df

def signals(df):
    df['Strong_Buy'] = (
        (df['20D_Z_SCORE_CLOSE'] <= -2.1) | 
        (
            (df['di_diff_20D_zscore'] <= -2) & 
            (df['20D_Z_SCORE_CLOSE'] <= -2)
        )
    )
    df['Strong_Sell'] = (
        (df['20D_Z_SCORE_OPEN'] >= 2.2) | 
        (
            (df['di_diff_20D_zscore'] >= 2) & 
            (df['20D_Z_SCORE_OPEN'] >=2)
        )
    )

    return df

def trade_book(depth = 120): # How long back you want look, 
    # this will book trades for that period and start recording open positions from next day onwards.  
    latest_date = df['date'].max()
    cutoff_date = latest_date - timedelta(days=depth)

    recent_data = df[df['date'] >= cutoff_date]
    
    symbols = recent_data['symbol'].unique()
    trades = []
    open_positions = []

    for symbol in symbols:
        symbol_df = recent_data[recent_data['symbol'] == symbol].sort_values('date', ascending = True)
        for _, row in symbol_df.iterrows():
            if not open_positions and row ['Strong_Buy'] == 1 and row['date'] < latest_date:
                trade = {
                    'symbol' : symbol,
                    'entry_date' : row['date'],
                    'entry_price' : row['next_open']
                }
                trades.append(trade)
                open_positions.append(trade)
            elif open_positions and row ['Strong_Sell'] == 1 and row['date'] < latest_date:
                trade.update({'exit_price' : row['next_open']})
                open_positions.pop()


        #Function definitions end here ----------------------------------
df = fetch_nse_data()
adx_prop = df.groupby('symbol', group_keys=False).apply(lambda g: calc_adx(g, window=15)) #Using proprietory function 

df = df.merge(adx_prop, on=['symbol', 'date'], how='left')

#setting date time format
df['date'] = pd.to_datetime(df['date'])
df['next_open'] = df.groupby('symbol')['open'].shift(-1)
 

df = df.sort_values(by=['symbol', 'date'], ascending=[True, True])  # Very important!

#Stop loss only works for open positions.

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

