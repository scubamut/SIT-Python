import pandas as pd
import time, datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pytz
from finhelpers import *
plt.rc('figure', figsize=(15, 10))

#*****************************************************************
# Load historical data
#****************************************************************** 

tickers = ['SPY','EFA','EWJ','EEM','IYR','RWX','IEF','TLT','DBC','GLD']
#tickers = [ 'SHY', 'CSD', 'IJR', 'MDY', 'PBE', 'GURU', 'EFA', 'EPP', 'EWA', 'IEV', 'ADRE', 'DEM', 'RWO', 'RWX', 'GLD']
#tickers = ['SPY','EFA','EWJ']
data_path = 'G:\\Google Drive\\Python Projects\\DATA\\'
start_date = dt.datetime(2004,12,1)
end_date = dt.datetime(2012,8,31)

try :
    data = pd.read_pickle(data_path + 'Equal_Weight.pkl')
except :
    # this uses finlib to load data
    data = get_history(tickers, start_date, end_date, data_path)
    data.major_axis = data.major_axis.tz_localize(pytz.utc)
    data.minor_axis = np.array(['open', 'high', 'low', 'close', 'volume', 'price'], dtype=object)
    data.to_pickle(data_path + 'Equal_Weight.pkl')
    
data_prices = data.ix[:,:,'price']

#*****************************************************************
# Equal Weight
#******************************************************************
capital = 100000.
prices = data_prices.copy()
n = len(prices.columns)

# find period ends
period_ends = endpoints(start_date, end_date, 'm')

weights = pd.DataFrame([[1. / prices.ix[date].count()] * n for date in period_ends],\
                       index=period_ends, columns=prices.columns)
returns = prices.pct_change()
p_value, p_holdings, p_returns, p_weights = backtest(prices, weights, period_ends, capital, offset=1., commission=0.)
p_value.plot()
plt.title('EQUAL WEIGHT PORTFOLIO')
plt.show()

print_stats(p_value)

m_rets = (1 + p_returns).resample('M', how='prod', kind='period') - 1
table = Monthly_Return_Table(m_rets)
print table


