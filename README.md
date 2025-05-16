# meanReversion

Using the theory that prices eventually cross their mean prices. I have built this strategy.

Using 20 Day Z Scores, RSI, ADX and DI_DIFF = +DI - (-DI) and their 20D z score based on 14D ADX calculations. 

The Z score helps determine possible chances of fallback/rise to mean. While other indicators are used solely to check momentum in an attempt to ride the up-trend and surf the down-trend wave.

This is buy only strategy with no consecutive buys.

update_trade_holidays.py : Knowing about the trade days available in a year is extremely imp. for setting cooldown period. Hence trade holiday data is manually fetched and entered into the function. 


This project implements a **mean reversion-based signal generation system** for equities using technical indicators. It processes market data from a PostgreSQL `nsedata_fact` table, computes various indicators, and outputs **Buy/Sell signals** based on statistically filtered logic. These signals can be plugged into any backtesting or execution system.

---

## 🧠 Strategy Logic

The strategy identifies reversion opportunities based on:
- **Z-score deviation** from 20-day Exponential Moving Average (EMA)
- **RSI (Relative Strength Index)** extremes
- **ADX-based Directional Momentum Filter** using:
  - +DI / -DI values
  - `DI_DIFF = +DI - -DI`
  - `DI_DIFF_SLOPE` (momentum of DI_DIFF)
- **Cooldown windows** and **quantile-based trend suppression filters** to avoid high-momentum periods

---

## 🔁 Workflow

1. **Input**: Historical price data from `nsedata_fact` (PostgreSQL)
2. **Feature Engineering**:
   - EMA, RSI, ADX, +DI, -DI
   - DI_DIFF and its slope
   - Cooldown period tracking to avoid overtrading
   - Volume based liquidity check
3. **Signal Logic**:
   - Entry signal when price is statistically far from EMA (Z-score)
   - Confirmed by RSI and DI momentum filters
   - Trend suppression logic avoids trades in strong trending environments
4. **Output**: DataFrame with timestamped Buy/Sell signals and associated metadata

---

## ⚙️ Dependencies

- **Python**
- **PostgreSQL** (`psycopg2`, `sqlalchemy`)
- **Pandas / Numpy**
- **TA-Lib / Custom TA functions**
- **Scikit-learn / Scipy** (for Z-score and quantile filters)

---

Internally Uses:

<img width="661" alt="Screenshot 2025-05-16 at 1 02 31 PM" src="https://github.com/user-attachments/assets/c3236f7a-d852-486d-9a05-c2a7f6788011" />

Backtesting Results:

<img width="303" alt="Screenshot 2025-05-16 at 1 09 40 PM" src="https://github.com/user-attachments/assets/dde6fd64-ee81-4688-b79a-680897a1f4bb" />

Author:

Dhruv Khatri
dhruvkhatri9275@gmail.com




