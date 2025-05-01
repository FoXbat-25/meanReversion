import sys
sys.path.append('/home/sierra1/projects/meanReversion')

from config import SQL_POSTGRES_CONN, SQL_ALCHEMY_CONN

import psycopg2
import pandas as pd 
import numpy as np
from datetime import datetime, timedelta

from utils import utils

conn = psycopg2.connect(SQL_POSTGRES_CONN)
cursor = conn.cursor()

def trade_book(df, depth:bool = False):
    
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
        SET exit_date = %s, exit_price = %s, updated_at = %s, status = 'closed', stp_lss_trggrd = True, cooldown_end_date = %s
        WHERE symbol = %s AND status = 'open'
    """

    latest_trade_date = df['prev_date'].max()
        
    if depth == False:
        recent_data = df[df['date'] == latest_trade_date]
    
    else:
        recent_data = df[df['date']<=latest_trade_date]

    symbols = recent_data['symbol'].unique()

    for symbol in symbols:
        
        symbol_df = recent_data[recent_data['symbol'] == symbol].sort_values('date', ascending = True)
        
        cooldown_end_date = None
        open_positions = None
        entry_atr = None

        for _, row in symbol_df.iterrows():

                if cooldown_end_date is not None and row['date'] >= cooldown_end_date:
                    cooldown_end_date = None                    
            
                if open_positions is None and row ['Strong_Buy'] == 1 and cooldown_end_date is None:
                    
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
                    entry_atr = (row['ATR'])

                elif open_positions is not None and row ['Strong_Sell'] == 1:
                    now_timestamp = datetime.now()

                    exit_trade_data = [row['next_date'],
                                row['next_open'],
                                now_timestamp,
                                row['symbol']]
                    
                    cursor.execute(exit_insert_query, tuple(exit_trade_data))
                    open_positions = None
                    entry_atr = None
                
                elif open_positions is not None and (
                                    
                    (row['close'] <= (open_positions - (1.5*entry_atr)))
        
                    ):

                    now_timestamp = datetime.now()
                    stop_loss_exit_data = [row['next_date'],
                                    row['next_open'],
                                    now_timestamp,
                                    row['cooldown_end_date'],
                                    row['symbol']
                                    ]
                    cursor.execute(stop_loss_exit_query, tuple(stop_loss_exit_data))
                    open_positions = None
                    entry_atr = None
                    cooldown_end_date = row['cooldown_end_date']

    
    conn.commit()
    cursor.close()
    conn.close()

    return df