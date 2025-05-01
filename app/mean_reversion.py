from utils import utils
from trade_book_copy import trade_book

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

def main():

    utils_obj = utils(start_date='2024-01-01')

    df = utils_obj.fetch_nse_data()
    df = utils_obj.fetch_cooldown_end_date(df)
       
    df = utils_obj.calc_adx(df, window=15)   
    df = utils_obj.calc_z_score(df)
    df = utils_obj.volume_check(df)
    df = utils_obj.calc_di_diff(df)
    df = strategy(df)  
    trade_book(df, depth = True)  #Only accepts True/False

    
    # print(df[df["symbol"] == "3IINFOLTD"].tail(20))
    df_first_row=df.groupby('symbol').head(1)
    print(df_first_row)

if __name__ == "__main__":
    main()
 