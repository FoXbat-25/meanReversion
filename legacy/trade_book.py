import sys
sys.path.append('/home/sierra1/projects/meanReversion')

from config import SQL_POSTGRES_CONN, SQL_ALCHEMY_CONN

import psycopg2
import pandas as pd 
import numpy as np
from datetime import datetime, timedelta

def trade_book(df, depth = None): # How long back you want look, 
    
    conn = psycopg2.connect(SQL_POSTGRES_CONN)
    cursor = conn.cursor()

    latest_date = df['date'].max()
    
    if depth is not None:
        cutoff_date = latest_date - timedelta(days=depth)
        recent_data = df[df['date'] >= cutoff_date]
    else:
        recent_data = df

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

    cooldown_fetch_query = """
        SELECT date, cooldown_end_date
        FROM CALENDAR c
        WHERE holiday = False
        ORDER BY date ASC;
    """
    cursor.execute(cooldown_fetch_query)
    cooldown_results = cursor.fetchall()

    # Create a symbol-to-date mapping
    cooldown_dict = {symbol: cooldown_date for symbol, cooldown_date in cooldown_results}

    symbols = recent_data['symbol'].unique()
 

    for symbol in symbols:
        symbol_df = recent_data[recent_data['symbol'] == symbol].sort_values('date', ascending = True)
        open_positions = None
        entry_atr = None
        for _, row in symbol_df.iterrows():
            
            if open_positions is None and row ['Strong_Buy'] == 1 and row['date'] < latest_date:
                now_timestamp = datetime.now()
                
                cooldown_date = cooldown_dict.get(row['date'])
                if cooldown_date and row['next_date'] < cooldown_date:
                    continue

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

                    
            elif open_positions is not None and row ['Strong_Sell'] == 1 and row['date'] < latest_date:
                now_timestamp = datetime.now()

                exit_trade_data = [row['next_date'],
                            row['next_open'],
                            now_timestamp,
                            row['symbol']]
                
                cursor.execute(exit_insert_query, tuple(exit_trade_data))
                open_positions = None
                entry_atr = None

            elif open_positions is not None and row['date'] < latest_date and (
                                
                (row['close'] <= (open_positions - (1.5*entry_atr)))
    
                ):

                now_timestamp = datetime.now()
                stop_loss_exit_data = [row['next_date'],
                                row['next_open'],
                                now_timestamp,
                                row['symbol'],
                                ]
                cursor.execute(stop_loss_exit_query, tuple(stop_loss_exit_data))
                open_positions = None
                entry_atr = None

    conn.commit()
    cursor.close()
    conn.close()

    return df