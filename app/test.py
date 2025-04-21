import sys
sys.path.append('/home/sierra1/projects/meanReversion')
from config import SQL_ALCHEMY_CONN

import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(SQL_ALCHEMY_CONN)
symbol = 'TEMBO'
query=f"""
select * from NSEDATA_FACT where symbol = '{symbol}'
"""

df = pd.read_sql(query, engine)
df = df.sort_values(by=['date'], ascending=[False])
print(df)

query2=f"""
select * from metadata where split_date IS NOT NULL;
"""
df = pd.read_sql(query2, engine)
# df = df.sort_values(by=['date'], ascending=[False])
print(df)