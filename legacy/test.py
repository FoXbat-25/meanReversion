import sys
sys.path.append('/home/sierra1/projects/meanReversion')
from config import SQL_ALCHEMY_CONN, SQL_POSTGRES_CONN
import psycopg2

from ta.trend import ADXIndicator

import pandas as pd 
from sqlalchemy import create_engine
import numpy as np

from datetime import datetime, timedelta


engine = create_engine(SQL_ALCHEMY_CONN)

query = """
        SELECT f.SYMBOL, f.DATE, f.OPEN, f.HIGH, f.LOW, f.CLOSE, f.VOLUME
        FROM NSEDATA_FACT f
        INNER JOIN METADATA2 m ON f.SYMBOL = m.SYMBOL 
        WHERE m.LISTING_DATE <= CURRENT_DATE - INTERVAL '65 days'
        AND f.DATE >=  CURRENT_DATE - INTERVAL '150';
    """

df = pd.read_sql(query, engine)
symbols = df['symbol'].unique()

def trade_book(df, depth = 130): # How long back you want look, 
    
    next_day_query = """
        SELECT DATE FROM CALENDAR WHERE DATE > CURRENT_DATE AND HOLIDAY = FALSE ORDER BY DATE ASC LIMIT 1;    
    """

    entry_insert_query = f"""
        INSERT INTO TRADE_BOOK (symbol, entry_date, entry_price, status, strategy, created_at, updated_at)
        VALUES %s
        ON CONFLICT (symbol, entry_date)
        DO NOTHING
    """
    exit_insert_query = f"""
        UPDATE TRADE_BOOK
        SET exit_date = %s, exit_price = %s, updated_at = %s, status = 'closed'
        WHERE symbol = %s AND status = 'open'
    """

    latest_date = df['date'].max()
    cutoff_date = latest_date - timedelta(days=depth)
    recent_data = df[df['date'] >= cutoff_date]
    
    symbols = recent_data['symbol'].unique()

    next_day = pd.read_sql(next_day_query, engine).iloc[0,0]
    df['next_open'] = df.groupby('symbol')['open'].shift(-1)

    conn = psycopg2.connect(SQL_POSTGRES_CONN)
    cursor = conn.cursor()

    for symbol in symbols:
        symbol_df = recent_data[recent_data['symbol'] == symbol].sort_values('date', ascending = True)
        open_positions = []
        for _, row in symbol_df.iterrows():
            
            if not open_positions and row ['Strong_Buy'] == 1 and row['date'] < latest_date:
                now_timestamp = datetime.now().timestamp()
                
                entry_trade_data = [row['symbol'],
                              next_day, 
                              row['next_open'], 
                              'open', 
                              'Mean Reversion', 
                              now_timestamp, 
                              now_timestamp ]
                
                cursor.execute(entry_insert_query, tuple(entry_trade_data))
                open_positions.append(next_day)

                    
            elif open_positions and row ['Strong_Sell'] == 1 and row['date'] < latest_date:
                now_timestamp = datetime.now().timestamp()

                exit_trade_data = [next_day,
                              row['next_open'],
                              now_timestamp,
                              row['symbol']]
                
                cursor.execute(exit_insert_query, tuple(exit_trade_data))
                open_positions = False

    conn.commit()
    cursor.close()
    conn.close()

