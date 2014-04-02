# Inspired by http://systematicinvestor.wordpress.com/2012/08/14/adaptive-asset-allocation/

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
    data = pd.read_pickle(data_path + 'framework_test.pkl')
except :
    # this uses finlib to load data
    data = get_history(tickers, start_date, end_date, data_path)
    data.major_axis = data.major_axis.tz_localize(pytz.utc)
    data.minor_axis = np.array(['open', 'high', 'low', 'close', 'volume', 'price'], dtype=object)
    data.to_pickle(data_path + 'framework_test.pkl')
    
data_prices = data.ix[:,:,'price']

#*****************************************************************
# Code Strategies
#******************************************************************
capital = 100000.
prices = data_prices.copy()
n = len(prices.columns)

# can have several portfolios
portfolios = {}

# find period ends
period_ends = endpoints(start_date, end_date, 'm')

# Adaptive Asset Allocation parameters
n_top = 5       # number of momentum positions
n_mom = 6 * 22    # length of momentum look back
n_vol = 1 * 22    # length of volatility look back 

#*****************************************************************
# Equal Weight
#******************************************************************

p_name = 'EqualWeightPortfolio'
portfolios = add_portfolio(p_name, portfolios)
weights = pd.DataFrame([[1. / prices.ix[date].count()] * n for date in period_ends],\
                       index=period_ends, columns=prices.columns)
returns = prices.pct_change()
p_value, p_holdings, p_returns, p_weights = backtest(prices, weights, period_ends, capital, offset=1., commission=0.)
p_value.plot()
plt.title(p_name)
plt.show()

print '\n\n {} \n\n'.format(p_name)
print_stats(p_value)

save_portfolio_metrics (portfolios, p_name, period_ends, prices, \
                            p_value, p_weights, path=None) 

#*****************************************************************
# Volatility Position Sizing
#******************************************************************

p_name = 'VolatilityPortfolio'
portfolios = add_portfolio(p_name, portfolios)

ret_log = np.log(1. + prices.pct_change())
hist_vol = pd.rolling_std(ret_log, n_vol) 
 
adj_vol = 1. / hist_vol.ix[period_ends,]
        
weights = (adj_vol / adj_vol.sum(axis=1)).ix[period_ends]    
p_value, p_holdings, p_returns, p_weights = backtest(prices, weights, period_ends, capital, offset=1., commission=0.)
p_value.plot()
plt.title(p_name)
plt.show()

print '\n\n {} \n\n'.format(p_name)
print_stats(p_value)
save_portfolio_metrics (portfolios, p_name, period_ends, prices, \
                            p_value, p_weights, path=None) 

#*****************************************************************
# Momentum Portfolio
#*****************************************************************

p_name = 'MomentumPortfolio'
portfolios = add_portfolio(p_name, portfolios)

momentum = prices / prices.shift(n_mom)
p = momentum.ix[period_ends]
rankings = p.rank(axis=1, ascending=False)

weights = ntop(rankings, n_top)
p_value, p_holdings, p_returns, p_weights = backtest(prices, weights, period_ends, capital, offset=1., commission=0.)
p_value.plot()
plt.title(p_name)
plt.show()

print '\n\n {} \n\n'.format(p_name)
print_stats(p_value)

save_portfolio_metrics (portfolios, p_name, period_ends, prices, \
                            p_value, p_weights, path=None) 

#***************************************************************************
# Combo: weight positions in the Momentum Portfolio according to Volatliliy
#***************************************************************************

p_name = 'Combo'
portfolios = add_portfolio(p_name, portfolios)

momentum = prices / prices.shift(n_mom)
p = momentum.ix[period_ends]
rankings = p.rank(axis=1, ascending=False)

w = ntop(rankings, n_top) * adj_vol
weights = w / (w.sum(axis=1))
p_value, p_holdings, p_returns, p_weights = backtest(prices, weights, period_ends, capital, offset=1., commission=0.)
p_value.plot()
plt.title(p_name)
plt.show()

print '\n\n {} \n\n'.format(p_name)
print_stats(p_value)

save_portfolio_metrics (portfolios, p_name, period_ends, prices, \
                            p_value, p_weights, path=None)

#*****************************************************************   
# Adaptive Asset Allocation (AAA)
# weight positions in the Momentum Portfolio according to 
# the minimum variance aligorithm
#***************************************************************** 

p_name = 'AdaptiveAssetAllocation'
portfolios = add_portfolio(p_name, portfolios)

momentum = prices / prices.shift(n_mom)
p = momentum.ix[period_ends]
rankings = p.rank(axis=1, ascending=False)

weights = ntop(rankings, n_top)

index = weights.index[weights.index==period_ends]
w_mv = pd.DataFrame(0., index=index, columns=weights.columns)
for i in range(len(period_ends) - 1) :
    if weights.ix[i].sum(1) != 0 :
        symbols = [column for column in weights.columns if weights[column].ix[i] > 0 ]
        idx = ret_log.index.searchsorted(period_ends[i])
        hist = ret_log.ix[idx - n_vol + 1 :idx + 1][symbols]
        
        ia = create_historical_ia(symbols, hist)
    
        s0 = ia['std_deviation']
        mu_vec = pd.DataFrame(ia['expected_return'])
        sigma_mat = ia['correlation'] * pd.DataFrame(s0).dot(pd.DataFrame(s0).T)
        
        mean=mu_vec.values
        lB=np.array([[0. for j in range(len(mu_vec))]]).T
        uB=np.array([[1. for j in range(len(mu_vec))]]).T
        covar=sigma_mat.values
#        print covar
        
        cla=CLA(mean,covar,lB,uB)
        cla.solve()
        w_mv.ix[i]=(w_mv.ix[i] + pd.Series(cla.getMinVar()[1].T[0], index=symbols).T).fillna(0)
#        print w_mv.index[i], '\n', pd.Series(cla.getMinVar()[1].T[0], index=symbols).T


p_value, p_holdings, p_returns, p_weights = backtest(prices, w_mv, period_ends, capital, offset=1., commission=0.)
p_value.plot()
plt.title(p_name)
plt.show()

print '\n\n {} \n\n'.format(p_name)
print_stats(p_value)
save_portfolio_metrics (portfolios, p_name, period_ends, prices, \
                            p_value, p_weights, path=None)

m_rets = (1 + p_returns).resample('M', how='prod', kind='period') - 1
table = Monthly_Return_Table(m_rets)
print table

# comparison table
metrics = ['cagr', 'sharpe', 'DVR', 'volatility', 'max_drawdown', 'avg_drawdown', 'VAR', 'CVAR']
table = pd.DataFrame(0., index=metrics, columns=portfolios.keys())
for name in portfolios.keys():
    for metric in metrics:
        table[name][metric] = portfolios[name][metric]

print table
