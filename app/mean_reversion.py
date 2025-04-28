from utils import utils
from trade_book import trade_book

import numpy as np
import pandas as pd

def strategy(df):

    df['Strong_Buy'] = (
        (df['20d_z_score_close'] <= -2.1) | 
        (
            (df['di_diff_20D_zscore'] <= -2) & 
            (df['20d_z_score_close'] <= -2)
        )
    )
    df['Strong_Sell'] = (
        (df['20d_z_score_open'] >= 2.2) | 
        (
            (df['di_diff_20D_zscore'] >= 2) & 
            (df['20d_z_score_open'] >=2)
        )
    )

    return df

def main():

    utils_obj = utils()

    df = utils_obj.fetch_nse_data()

    df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values(by=['symbol', 'date'], ascending=[True, True])
    

    utils_obj.calc_rsi(df)


    adx_prop = df.groupby('symbol', group_keys=False).apply(lambda g: utils_obj.calc_adx(g, window=15)) #Using proprietory function 
    df = df.merge(adx_prop, on=['symbol', 'date'], how='left')


    utils_obj.calc_di_diff(df)
    utils_obj.calc_z_score(df)
    strategy(df) 
    trade_book(df, depth =130) 

    # df = df.sort_values(by=['symbol', 'date'], ascending=[True, False])
    
    # print(df[df["symbol"] == "3IINFOLTD"].tail(20))
    # df_first_row=df.groupby('symbol').head(1)
    # print(df_first_row)

if __name__ == "__main__":
    main()
 