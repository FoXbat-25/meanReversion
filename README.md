# meanReversion

Using the theory that prices eventually cross their mean prices. I have built this strategy.

Using 20 Day Z Scores, RSI, ADX and DI_DIFF = +DI - (-DI) and their 20D z score based on 14D ADX calculations. 

The Z score helps determine possible chances of fallback/rise to mean. While other indicators are used solely to check momentum in an attempt to ride the up-trend and surf the down-trend wave.

This is buy only strategy with no consecutive buys.

update_trade_holidays.py : Knowing about the trade days available in a year is extremely imp. for setting cooldown period. Hence trade holiday data is manually fetched and entered into the function. 