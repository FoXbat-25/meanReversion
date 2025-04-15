import sys
sys.path.append('/home/sierra1/projects/meanReversion')
from config import SQL_ALCHEMY_CONN

import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(SQL_ALCHEMY_CONN)

query="""
select f
"""