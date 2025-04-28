import sys
sys.path.append('/home/sierra1/projects/meanReversion')

from config import SQL_POSTGRES_CONN

import psycopg2
import pandas as pd 
import numpy as np
from datetime import datetime, timedelta





def trade_book(df, depth = 130): # How long back you want look, 
    
    entry_insert_query = f"""
        INSERT INTO TRADE_BOOK (symbol, entry_date, entry_price, status, strategy, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, entry_date)
        DO NOTHING
    """
    exit_insert_query = f"""
        UPDATE TRADE_BOOK
        SET exit_date = %s, exit_price = %s, updated_at = %s, status = 'closed'
        WHERE symbol = %s AND status = 'open'
    """
    stop_loss_exit_query = f"""
        UPDATE TRADE_BOOK
        SET exit_date = %s, exit_price = %s, updated_at = %s, status = 'closed', stp_lss_trggrd = True
        WHERE symbol = %s AND status = 'open'
    """

    df['next_open'] = df.groupby('symbol')['open'].shift(-1)
    df['next_date'] = df.groupby('symbol')['date'].shift(-1) 

    latest_date = df['date'].max()
    cutoff_date = latest_date - timedelta(days=depth)
    recent_data = df[df['date'] >= cutoff_date]

    symbols = recent_data['symbol'].unique()

    conn = psycopg2.connect(SQL_POSTGRES_CONN)
    cursor = conn.cursor()

    for symbol in symbols:
        symbol_df = recent_data[recent_data['symbol'] == symbol].sort_values('date', ascending = True)
        open_positions = None
        for _, row in symbol_df.iterrows():
            
            if open_positions is None and row ['Strong_Buy'] == 1 and row['date'] < latest_date:
                now_timestamp = datetime.now()
                
                entry_trade_data = [row['symbol'],
                            row['next_date'], 
                            row['next_open'], 
                            'open', 
                            'Mean Reversion', 
                            now_timestamp, 
                            now_timestamp ]
                
                cursor.execute(entry_insert_query, tuple(entry_trade_data))
                open_positions = (row['next_open'])

                    
            elif open_positions is not None and row ['Strong_Sell'] == 1 and row['date'] < latest_date:
                now_timestamp = datetime.now()

                exit_trade_data = [row['next_date'],
                            row['next_open'],
                            now_timestamp,
                            row['symbol']]
                
                cursor.execute(exit_insert_query, tuple(exit_trade_data))
                open_positions = None

            elif open_positions is not None and row['date'] < latest_date and (
                                
                (row['close'] <= (open_positions - (1.5*row['ATR'])))
                or
                (row['20d_z_score_close'] <= -1.9 and row['di_diff_20D_zscore'] <= -1.9 and row['close'] <= 0.98*open_positions)
            ):
                now_timestamp = datetime.now()
                stop_loss_exit_data = [row['next_date'],
                                row['next_open'],
                                now_timestamp,
                                row['symbol'],
                                ]
                cursor.execute(stop_loss_exit_query, tuple(stop_loss_exit_data))
                open_positions = None

    conn.commit()
    cursor.close()
    conn.close()

    return df