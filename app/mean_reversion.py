import sys
import os
mean_reversion_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(mean_reversion_path)

from meanReversion.utils.utils import utils

import numpy as np
import pandas as pd

def strategy(df):

    df['Strong_Buy'] = (
        (
            # (df['20d_z_score_close'] <= -2.1) | 
            (
                (df['di_diff_20D_zscore'] <= -2) & 
                (df['20d_z_score_close'] <= -2) &
                (df['RSI_7'] < 30)
            )
        ) & (df['volume_flag'] == True)
    )

    df['Strong_Sell'] = (
        # (df['20d_z_score_open'] >= 2.2) | 
        (
            (df['di_diff_20D_zscore'] >= 2) & 
            (df['20d_z_score_close'] >=2) &
            (df['RSI_7'] >75)
        )
    )

    return df

def mean_reversion(start_from='2023-01-01', adx_window=15,atr_window=14, z_score_window_list=[20], min_volume=500000, vol_window = 20, di_diff_window=20):

    utils_obj = utils(start_date=start_from)
    
    df = utils_obj.dataframe
    df = utils_obj.calc_adx(df, adx_window=adx_window,atr_window=atr_window)
    df = utils_obj.fetch_cooldown_end_date(df)
    df = utils_obj.calc_rsi(df)   
    df = utils_obj.calc_z_score(df, span=z_score_window_list) # span accepts only list
    df = utils_obj.volume_check(df, min_volume=min_volume, rolling_window=vol_window)
    df = utils_obj.calc_di_diff(df,rolling_window=di_diff_window)
    
    df = strategy(df)
    df = df[df['date'] >= start_from] 
    # df = df[((df['RSI_7'] < 25) | (df['RSI_7'] >75)) & ((df['20d_z_score_close'] > 2) | (df['20d_z_score_close'] < -2)) & ((df['20d_z_score_close'] <= -2) | (df['di_diff_20D_zscore'] >= 2))] 
    # pd.set_option('display.max_columns', None)
    # print(df)

    # df = df.sort_values(by=['symbol', 'date'], ascending=[True, False])

    # pd.set_option('display.max_columns', None)

    # print(df[df["symbol"] == "KPIL"])

    # print(df[df['cooldown_end_date'].isna()])

    
    # df_first_row=df.groupby('symbol').head(1)
    # print(df_first_row)

    return df

if __name__ == "__main__":
    mean_reversion()
 