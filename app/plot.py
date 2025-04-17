import sys
sys.path.append('/home/sierra1/projects/meanReversion')

import matplotlib as plt
from mean_reversion import df_first_symbol

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# Subplot: ADX +DI -DI + Close price
plt.subplot(2, 1, 1)
plt.plot(df_first_symbol['date'], df_first_symbol['close'], label='Close', color='black', alpha=0.7)
#plt.plot(df_first_symbol['date'], df_first_symbol['ADX'], label='ADX', color='blue', alpha=0.7)
#plt.plot(df_first_symbol['date'], df_first_symbol['+DI'], label='+DI', color='green', alpha=0.7)
#plt.plot(df_first_symbol['date'], df_first_symbol['-DI'], label='-DI', color='red', alpha=0.7)
plt.plot(df_first_symbol['date'], df_first_symbol['di_diff'], label='DI_DIFF', color='maroon', alpha=0.7)
plt.plot(df_first_symbol['date'], df_first_symbol['di_diff_slope'], label='DI_DIFF_SLOPE', color='black', alpha=0.7)


#plt.title('ADX, +DI, -DI, and Close Price')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("plot.png", bbox_inches='tight')
plt.close()

