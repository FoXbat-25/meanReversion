import sys
sys.path.append('/home/sierra1/projects/meanReversion')

from meanReversion.utils.utils import utils
from backtestingEngine.app.trade_book_copy import trade_book

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

def strategy(df):

    df['Strong_Buy'] = (
        (
            (df['20d_z_score_close'] <= -2.1) | 
            (
                (df['di_diff_20D_zscore'] <= -2) & 
                (df['20d_z_score_close'] <= -2)
            )
        ) & (df['volume_flag'] == True)
    )

    df['Strong_Sell'] = (
        (df['20d_z_score_open'] >= 2.2) | 
        (
            (df['di_diff_20D_zscore'] >= 2) & 
            (df['20d_z_score_close'] >=2)
        )
    )

    return df

def mean_reversion(adx_window=15, z_score_window_list=[20, 45], min_volume=500000, vol_window = 20, di_diff_window=20):

    utils_obj = utils()

    df = utils_obj.fetch_nse_data()
    df = utils_obj.fetch_cooldown_end_date(df)
    df = utils_obj.calc_adx(df, window=adx_window)   
    df = utils_obj.calc_z_score(df, span=z_score_window_list) # span accepts only list
    df = utils_obj.volume_check(df, min_volume=min_volume, rolling_window=vol_window)
    df = utils_obj.calc_di_diff(df,rolling_window=di_diff_window)
    df = strategy(df)  
     #Only accepts True/False

    
    # print(df[df["symbol"] == "3IINFOLTD"].tail(20))
    df_first_row=df.groupby('symbol').head(1)
    print(df_first_row)

    return df

if __name__ == "__main__":
    mean_reversion()
 