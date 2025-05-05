import sys
sys.path.append('/home/sierra1/projects/meanReversion')
from app.config import SQL_POSTGRES_CONN

import pandas as pd
import psycopg2

#Will have to manually get holiday list for each year and feed into the function.

csv_file = '../data/NSE trading holidays 2023.csv'
def trade_date(): #This is the next day after a signal has been provided.
        calendar_df = pd.read_csv(csv_file, parse_dates=['date'], dayfirst=False)
        calendar_df['remarks'] = calendar_df['remarks'].fillna('')
        calendar_df['holiday'] = True
        data = list(calendar_df[['date', 'day', 'remarks', 'holiday']].itertuples(index=False, name=None))
        # print(data)
        update_query = """
        UPDATE CALENDAR_COPY
        SET
            remarks = %s,
            holiday = %s
        WHERE
            date = %s
        """

        conn = psycopg2.connect(SQL_POSTGRES_CONN)
        cursor = conn.cursor()

        # Notice we pass (remarks, holiday, date) in that order!
        formatted_data = [(d[2], d[3], d[0]) for d in data]

        for row in formatted_data:
            cursor.execute(update_query, row)
        conn.commit()

        cursor.close()
        conn.close()

        return None


if __name__ == "__main__":
        trade_date()