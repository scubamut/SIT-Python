import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.io.data as web
from math import sqrt
import math


#############################
# FINANCIAL HELPER ROUTINES #
#############################

def compute_nyears(x) :
    return np.double((x.index[-1] - x.index[0]).days) / 365.

def compute_cagr(equity) :
    return np.double((equity.ix[-1] / equity.ix[0]) ** (1. / compute_nyears(equity)) - 1)

def compute_annual_factor(equity) :
    possible_values = [252,52,26,13,12,6,4,3,2,1]
    L = pd.Series(len(equity) / compute_nyears(equity) - possible_values)
    return possible_values[L.index[L == L.min()]]

def compute_sharpe(equity) :
    rets = equity / equity.shift(1) - 1
    temp = compute_annual_factor(rets)
    rets = pd.Series(rets)
    return sqrt(temp) * rets.mean()/rets.std()

def compute_DVR(equity):
    return compute_sharpe(equity) * compute_R2(equity) 

def compute_drawdown(x) :
    return (x - pd.expanding_max(x))/pd.expanding_max(x)

def compute_max_drawdown(x):
    return compute_drawdown(x).min()

def compute_rolling_drawdown(equity) :
    rolling_dd = pd.rolling_apply(equity, 252, compute_max_drawdown, min_periods=0)
    df = pd.concat([equity, rolling_dd], axis=1)
    df.columns = ['x', 'rol_dd_10']
    plt.plot(df)
    plt.grid()

def compute_avg_drawdown(x) :
    drawdown = compute_drawdown(x).shift(-1)
    drawdown[-1]=0.
    dend = [drawdown.index[i] for i in range(len(drawdown)) if drawdown[i] == 0 and drawdown[i-1] != 0]
    dstart = [drawdown.index[i] for i in range(len(drawdown)-1) if drawdown[i] == 0 and drawdown[i+1] != 0]
    f = pd.DataFrame([dstart, dend], index=['dstart', 'dend']).transpose()
    f['drawdown'] = [drawdown[f['dstart'][i]:f['dend'][i]].min() for i in range(len(f))]
    return f.drawdown.mean()

def compute_calmar(x) :
    return compute_cagr(x) / compute_max_drawdown(x)

def compute_R2(equity) :
    x = pd.DataFrame(equity)
    x.columns=[0]
    x[1]=[equity.index[i].toordinal() for i in range(len(equity))]
    return x[0].corr(x[1]) ** 2

def compute_volatility(x) :
    temp = compute_annual_factor(x)
    return sqrt(temp) * x.std()

def compute_var(x, probs=0.05) :
    return x.quantile(probs)

def compute_cvar(x, probs=0.05) :
    return x[ x < x.quantile(probs) ].mean()

def print_stats(equity) :
    print '**** STATISTICS ****'
    print '====================\n'
    print 'n_years        : ', compute_nyears(equity)
    print 'cagr        : ', compute_cagr(equity) * 100, '%'
    rets = equity / equity.shift(1) - 1
    print 'annual_factor    : ', compute_annual_factor(equity)
    print 'sharpe        : ', compute_sharpe(equity)
    compute_drawdown(rets)
    print 'max_drawdown    : ', compute_max_drawdown(equity) * 100, '%'
    print 'avg_drawdown    : ', compute_avg_drawdown(equity) * 100, '%'
    print 'calmar        : ', compute_calmar(equity)
    print 'R-squared    : ', compute_R2(equity)
    print 'DVR        : ', compute_DVR(equity)
    print 'volatility    : ', compute_volatility(rets)
    #print 'exposure    : ', compute_exposure(models$equal_weight)
    print 'VAR 5%       : ', compute_var(equity)
    print 'CVAR 5%       : ', compute_cvar(equity)
    
from zipline.utils import tradingcalendar

def endpoints(start, end, period='m') :
    
    dates = tradingcalendar.get_trading_days(start, end)

    if isinstance(period, int) :
        dates = [dates[i] for i in range(0, len(dates), period)]
    else :    
        if period == 'm' : months = 1
        elif period == 'q' : months = 3
        elif period == 'b' : months = 6
        elif period == 'y' : months = 12           
            
        e_dates = [dates[i - 1] for i in range(1,len(dates))\
                          if dates[i].month > dates[i-1].month\
                          or dates[i].year > dates[i-1].year ]+ list([dates[-1]])
        dates = [e_dates[i] for i in range(0,len(e_dates),months)]
    
    return dates

def add_portfolio(name, portfolios) :
    return dict(portfolios.items() + {name : {}}.items())

# topn 
def ntop(prices, n) :
    weights = pd.DataFrame(0., index=prices.index, columns=prices.columns)
    for i in range(len(prices)) :
        n_not_na = prices.ix[i].count()
        n_row = min(n, n_not_na) 
        for s in prices.columns :
            if prices.ix[i][s] <= n :
                weights.ix[i][s] = 1. / n_row
            else :
                weights.ix[i][s] = 0.
    
    return weights

def Monthly_Return_Table (monthly_returns) :
    # monthly_returns is a pandas Series of monthly returns indexed by date(Y-m)
    df = pd.DataFrame(monthly_returns.values, columns=['Data'])
    df['Month'] = monthly_returns.index.month
    df['Year']= monthly_returns.index.year
    table = df.pivot_table(rows='Year', cols='Month').fillna(0)
    table['Annual Return'] = table.apply(np.sum, axis=1) * 100
    
    return table
    
def generate_orders(transactions) :
    orders = pd.DataFrame()
    for i in range(len(transactions)):
        for j in range(len(transactions.columns)):
            t = transactions.ix[i]
            if transactions.ix[i][j] < 0 :
                orders = orders.append([[t.name.date().year, t.name.date().month, t.name.date().day, t.index[j], 'Sell', abs(t[j])]])
            if transactions.ix[i][j] > 0 :
                orders = orders.append([[t.name.date().year, t.name.date().month, t.name.date().day, t.index[j], 'Buy', abs(t[j])]])
    orders.columns = ['Year', 'Month', 'Day', 'Symbol', 'Action', 'Qty']
    
    return orders


def save_portfolio_metrics (portfolios, portfolio_name, period_ends, prices, \
                            p_value, p_weights, path=None) :
        
    rebalance_qtys = (p_weights.ix[period_ends] / prices.ix[period_ends]) * p_value.ix[period_ends]
    p_holdings = rebalance_qtys.align(prices)[0].shift(1).ffill().fillna(0)
    transactions = p_holdings - p_holdings.shift(1).fillna(0)
    
    p_returns = p_value.pct_change(periods=1)
    p_index = np.cumproduct(1 + p_returns)
    
    m_rets = (1 + p_returns).resample('M', how='prod', kind='period') - 1
    
    portfolios[portfolio_name]['equity'] = p_value
    portfolios[portfolio_name]['ret'] = p_returns
    portfolios[portfolio_name]['cagr'] = compute_cagr(p_value) * 100
    portfolios[portfolio_name]['sharpe'] = compute_sharpe(p_value)
    portfolios[portfolio_name]['weight'] = p_weights
    portfolios[portfolio_name]['transactions'] = transactions
    portfolios[portfolio_name]['period_return'] = 100 * (p_value.ix[-1] / p_value[0] - 1)
    portfolios[portfolio_name]['avg_monthly_return'] = p_index.resample('BM', how='last').pct_change().mean() * 100
    portfolios[portfolio_name]['monthly_return_table'] = Monthly_Return_Table(m_rets)
    portfolios[portfolio_name]['drawdowns'] = compute_drawdown(p_value).dropna()
    portfolios[portfolio_name]['max_drawdown'] = compute_max_drawdown(p_value) * 100
    portfolios[portfolio_name]['max_drawdown_date'] = p_value.index[compute_drawdown(p_value)==compute_max_drawdown(p_value)][0].date().isoformat()
    portfolios[portfolio_name]['avg_drawdown'] = compute_avg_drawdown(p_value) * 100
    portfolios[portfolio_name]['calmar'] = compute_calmar(p_value)
    portfolios[portfolio_name]['R_squared'] = compute_calmar(p_value)
    portfolios[portfolio_name]['DVR'] = compute_DVR(p_value)
    portfolios[portfolio_name]['volatility'] = compute_volatility(p_returns)
    portfolios[portfolio_name]['VAR'] = compute_var(p_value)
    portfolios[portfolio_name]['CVAR'] = compute_cvar(p_value)
    portfolios[portfolio_name]['rolling_annual_returns'] = pd.rolling_apply(p_returns, 252, np.sum) 
    portfolios[portfolio_name]['p_holdings'] = p_holdings
    portfolios[portfolio_name]['transactions'] = np.round(transactions[transactions.sum(1)!=0], 0)
    portfolios[portfolio_name]['share'] = p_holdings
    portfolios[portfolio_name]['orders'] = generate_orders(transactions)
    portfolios[portfolio_name]['best'] = max(p_returns)
    portfolios[portfolio_name]['worst'] = min(p_returns)

    if path != None :
        portfolios[portfolio_name].equity.to_csv(path + portfolio_name + '_equity.csv')
        portfolios[portfolio_name].weight.to_csv(path + portfolio_name + '_weight.csv')
        portfolios[portfolio_name].share.to_csv(path + portfolio_name + '_share.csv')
        portfolios[portfolio_name].transactions.to_csv(path + portfolio_name + '_transactions.csv')
        portfolios[portfolio_name].orders.to_csv(path + portfolio_name + '_orders.csv')
        
    return

def backtest(prices, weights, period_ends, capital, offset=0., commission=0.) : 
    
    p_holdings = (capital / prices * weights.align(prices)[0]).shift(offset).ffill().fillna(0)
    w = weights.align(prices)[0].shift(offset).fillna(0)
    trade_dates = w[w.sum(1) != 0].index
    p_cash = capital - (p_holdings * prices.shift(offset)).sum(1)
    totalcash = p_cash[trade_dates].align(prices[prices.columns[0]])[0].ffill().fillna(0)
    p_returns = (totalcash  + (p_holdings * prices).sum(1) - \
                    (abs(p_holdings - p_holdings.shift(1)) * commission).sum(1)) / \
                    (totalcash + (p_holdings * prices.shift(1)).sum(1)) - 1
    p_returns = p_returns.fillna(0)
#    p_weights = p_holdings * prices.shift(offset) / (totalcash + (p_holdings * prices.shift(offset)).sum(1))
    p_weights = pd.DataFrame([(p_holdings * prices.shift(offset))[symbol] / \
                              (totalcash + (p_holdings * prices.shift(offset)).sum(1)) \
                              for symbol in prices.columns], index=prices.columns).T
    p_weights = p_weights.fillna(0)

    return np.cumproduct(1. + p_returns) * capital, p_holdings, p_returns, p_weights

# note: hist_returns are CONTINUOUSLY COMPOUNDED RETURNS
# ie R = e ** hist_returns
def create_historical_ia(symbols, hist_returns, annual_factor=252) :
    
    ia = {}
    ia['n'] = len(symbols)
    ia['annual_factor'] = annual_factor
    ia['symbols'] = hist_returns.columns
    ia['symbol_names'] = hist_returns.columns
    ia['hist_returns'] = hist_returns[symbols]
#    ret = hist_returns[symbols].apply(lambda(x): (e ** x) -1.)
    ia['arithmetic_return'] = hist_returns[symbols].mean()
    ia['geometric_return'] = hist_returns[symbols].apply(lambda(x): np.prod(1. + x) ** (1. / len(x)) -1.)
    ia['std_deviation'] = hist_returns[symbols].std()
    ia['correlation'] = hist_returns[symbols].corr()
    ia['arithmetic_return'] = (1. + ia['arithmetic_return']) ** ia['annual_factor'] - 1.
    ia['geometric_return'] = (1. + ia['geometric_return']) ** ia['annual_factor'] - 1.
    ia['risk'] = sqrt(ia['annual_factor']) * ia['std_deviation']
    for i in range(len(ia['risk'])):
        if ia['risk'][i].round(6) == 0.0 : ia['risk'][i] = 0.0000001
    ia['cov'] = ia['correlation'] * (ia['risk'].dot(ia['risk'].T))
    ia['expected_return'] = ia['arithmetic_return']
    return(ia)

#p_value, p_holdings, p_returns, p_weights = backtest(prices, weights, period_ends, capital, offset=1., commission=0.)

iif = lambda a,b,c: (b,c)[not a]
def ifna(x,y) :
    return(iif(math.isnan(x)(x) or math.isinf(x), y, x))

#!/usr/bin/env python
# On 20130210, v0.2
# Critical Line Algorithm
# by MLdP <lopezdeprado@lbl.gov>

#---------------------------------------------------------------
#---------------------------------------------------------------
class CLA:
    def __init__(self,mean,covar,lB,uB):
        # Initialize the class
        self.mean=mean
        self.covar=covar
        self.lB=lB
        self.uB=uB
        self.w=[] # solution
        self.l=[] # lambdas
        self.g=[] # gammas
        self.f=[] # free weights
#---------------------------------------------------------------
    def solve(self):
        # Compute the turning points,free sets and weights
        f,w=self.initAlgo()
        self.w.append(np.copy(w)) # store solution
        self.l.append(None)
        self.g.append(None)
        self.f.append(f[:])
        while True:
            #1) case a): Bound one free weight
            l_in=None
            if len(f)>1:
                covarF,covarFB,meanF,wB=self.getMatrices(f)
                covarF_inv=np.linalg.inv(covarF)
                j=0
                for i in f:
                    l,bi=self.computeLambda(covarF_inv,covarFB,meanF,wB,j,[self.lB[i],self.uB[i]])
                    if l>l_in:l_in,i_in,bi_in=l,i,bi
                    j+=1
            #2) case b): Free one bounded weight
            l_out=None
            if len(f)<self.mean.shape[0]:
                b=self.getB(f)
                for i in b:
                    covarF,covarFB,meanF,wB=self.getMatrices(f+[i])
                    covarF_inv=np.linalg.inv(covarF)
                    l,bi=self.computeLambda(covarF_inv,covarFB,meanF,wB,meanF.shape[0]-1, \
                        self.w[-1][i])
                    if (self.l[-1]==None or l<self.l[-1]) and l>l_out:l_out,i_out=l,i                
            if (l_in==None or l_in<0) and (l_out==None or l_out<0):
                #3) compute minimum variance solution
                self.l.append(0)
                covarF,covarFB,meanF,wB=self.getMatrices(f)
                covarF_inv=np.linalg.inv(covarF)
                meanF=np.zeros(meanF.shape)
            else:
                #4) decide lambda
                if l_in>l_out:
                    self.l.append(l_in)
                    f.remove(i_in)
                    w[i_in]=bi_in # set value at the correct boundary
                else:
                    self.l.append(l_out)
                    f.append(i_out)
                covarF,covarFB,meanF,wB=self.getMatrices(f)
                covarF_inv=np.linalg.inv(covarF)
            #5) compute solution vector
            wF,g=self.computeW(covarF_inv,covarFB,meanF,wB)
            for i in range(len(f)):w[f[i]]=wF[i]
            self.w.append(np.copy(w)) # store solution
            self.g.append(g)
            self.f.append(f[:])
            if self.l[-1]==0:break
        #6) Purge turning points
        self.purgeNumErr(10e-10)
        self.purgeExcess()
#---------------------------------------------------------------    
    def initAlgo(self):
        # Initialize the algo
        #1) Form structured array
        a=np.zeros((self.mean.shape[0]),dtype=[('id',int),('mu',float)])
        b=[self.mean[i][0] for i in range(self.mean.shape[0])] # dump array into list
        a[:]=zip(range(self.mean.shape[0]),b) # fill structured array
        #2) Sort structured array
        b=np.sort(a,order='mu')
        #3) First free weight
        i,w=b.shape[0],np.copy(self.lB)
        while np.sum(w)<1:
            i-=1
            w[b[i][0]]=self.uB[b[i][0]]
        w[b[i][0]]+=1-np.sum(w)
        return [b[i][0]],w
#---------------------------------------------------------------    
    def computeBi(self,c,bi):
        if c>0:
            bi=bi[1][0]
        if c<0:
            bi=bi[0][0]
        return bi
#---------------------------------------------------------------
    def computeW(self,covarF_inv,covarFB,meanF,wB):
        #1) compute gamma
        onesF=np.ones(meanF.shape)
        g1=np.dot(np.dot(onesF.T,covarF_inv),meanF)
        g2=np.dot(np.dot(onesF.T,covarF_inv),onesF)
        if wB==None:
            g,w1=float(-self.l[-1]*g1/g2+1/g2),0
        else:
            onesB=np.ones(wB.shape)
            g3=np.dot(onesB.T,wB)
            g4=np.dot(covarF_inv,covarFB)
            w1=np.dot(g4,wB)
            g4=np.dot(onesF.T,w1)
            g=float(-self.l[-1]*g1/g2+(1-g3+g4)/g2)
        #2) compute weights
        w2=np.dot(covarF_inv,onesF)
        w3=np.dot(covarF_inv,meanF)
        return -w1+g*w2+self.l[-1]*w3,g
#---------------------------------------------------------------
    def computeLambda(self,covarF_inv,covarFB,meanF,wB,i,bi):
        #1) C
        onesF=np.ones(meanF.shape)
        c1=np.dot(np.dot(onesF.T,covarF_inv),onesF)
        c2=np.dot(covarF_inv,meanF)
        c3=np.dot(np.dot(onesF.T,covarF_inv),meanF)
        c4=np.dot(covarF_inv,onesF)
        c=-c1*c2[i]+c3*c4[i]
        if c==0:return None,None
        #2) bi
        if type(bi)==list:bi=self.computeBi(c,bi)
        #3) Lambda
        if wB==None:
            # All free assets
            return float((c4[i]-c1*bi)/c),bi
        else:
            onesB=np.ones(wB.shape)
            l1=np.dot(onesB.T,wB)
            l2=np.dot(covarF_inv,covarFB)
            l3=np.dot(l2,wB)
            l2=np.dot(onesF.T,l3)
            return float(((1-l1+l2)*c4[i]-c1*(bi+l3[i]))/c),bi
#---------------------------------------------------------------
    def getMatrices(self,f):
        # Slice covarF,covarFB,covarB,meanF,meanB,wF,wB
        covarF=self.reduceMatrix(self.covar,f,f)
        meanF=self.reduceMatrix(self.mean,f,[0])
        b=self.getB(f)
        covarFB=self.reduceMatrix(self.covar,f,b)
        wB=self.reduceMatrix(self.w[-1],b,[0])
        return covarF,covarFB,meanF,wB
#---------------------------------------------------------------
    def getB(self,f):
        return self.diffLists(range(self.mean.shape[0]),f)
#---------------------------------------------------------------
    def diffLists(self,list1,list2):
        return list(set(list1)-set(list2))
#---------------------------------------------------------------
    def reduceMatrix(self,matrix,listX,listY):
        # Reduce a matrix to the provided list of rows and columns
        if len(listX)==0 or len(listY)==0:return
        matrix_=matrix[:,listY[0]:listY[0]+1]
        for i in listY[1:]:
            a=matrix[:,i:i+1]
            matrix_=np.append(matrix_,a,1)
        matrix__=matrix_[listX[0]:listX[0]+1,:]
        for i in listX[1:]:
            a=matrix_[i:i+1,:]
            matrix__=np.append(matrix__,a,0)
        return matrix__
#---------------------------------------------------------------    
    def purgeNumErr(self,tol):
        # Purge violations of inequality constraints (associated with ill-conditioned covar matrix)
        i=0
        while True:
            flag=False
            if i==len(self.w):break
            if abs(np.sum(self.w[i])-1)>tol:
                flag=True
            else:
                for j in range(self.w[i].shape[0]):
                    if self.w[i][j]-self.lB[j]<-tol or self.w[i][j]-self.uB[j]>tol:
                        flag=True;break
            if flag==True:
                del self.w[i]
                del self.l[i]
                del self.g[i]
                del self.f[i]
            else:
                i+=1
        return
#---------------------------------------------------------------    
    def purgeExcess(self):
        # Remove violations of the convex hull
        i,repeat=0,False
        while True:
            if repeat==False:i+=1
            if i==len(self.w)-1:break
            w=self.w[i]
            mu=np.dot(w.T,self.mean)[0,0]
            j,repeat=i+1,False
            while True:
                if j==len(self.w):break
                w=self.w[j]
                mu_=np.dot(w.T,self.mean)[0,0]
                if mu<mu_:
                    del self.w[i]
                    del self.l[i]
                    del self.g[i]
                    del self.f[i]
                    repeat=True
                    break
                else:
                    j+=1
        return
#---------------------------------------------------------------
    def getMinVar(self):
        # Get the minimum variance solution
        var=[]
        for w in self.w:
            a=np.dot(np.dot(w.T,self.covar),w)
            var.append(a)
        return min(var)**.5,self.w[var.index(min(var))]
#---------------------------------------------------------------
    def getMaxSR(self):
        # Get the max Sharpe ratio portfolio
        #1) Compute the local max SR portfolio between any two neighbor turning points
        w_sr,sr=[],[]
        for i in range(len(self.w)-1):
            w0=np.copy(self.w[i])
            w1=np.copy(self.w[i+1])
            kargs={'minimum':False,'args':(w0,w1)}
            a,b=self.goldenSection(self.evalSR,0,1,**kargs)
            w_sr.append(a*w0+(1-a)*w1)
            sr.append(b)
        return max(sr),w_sr[sr.index(max(sr))]
#---------------------------------------------------------------
    def evalSR(self,a,w0,w1):
        # Evaluate SR of the portfolio within the convex combination
        w=a*w0+(1-a)*w1
        b=np.dot(w.T,self.mean)[0,0]
        c=np.dot(np.dot(w.T,self.covar),w)[0,0]**.5
        return b/c
#---------------------------------------------------------------
    def goldenSection(self,obj,a,b,**kargs):
        # Golden section method. Maximum if kargs['minimum']==False is passed 
        from math import log,ceil
        tol,sign,args=1.0e-9,1,None
        if 'minimum' in kargs and kargs['minimum']==False:sign=-1
        if 'args' in kargs:args=kargs['args']
        numIter=int(ceil(-2.078087*log(tol/abs(b-a))))
        r=0.618033989
        c=1.0-r
        # Initialize
        x1=r*a+c*b;x2=c*a+r*b
        f1=sign*obj(x1,*args);f2=sign*obj(x2,*args)
        # Loop
        for i in range(numIter):
            if f1>f2:
                a=x1
                x1=x2;f1=f2
                x2=c*a+r*b;f2=sign*obj(x2,*args)
            else:
                b=x2
                x2=x1;f2=f1
                x1=r*a+c*b;f1=sign*obj(x1,*args)
        if f1<f2:return x1,sign*f1
        else:return x2,sign*f2
#---------------------------------------------------------------
    def efFrontier(self,points):
        # Get the efficient frontier
        mu,sigma,weights=[],[],[]
        a=np.linspace(0,1,points/len(self.w))[:-1] # remove the 1, to avoid duplications
        b=range(len(self.w)-1)
        for i in b:
            w0,w1=self.w[i],self.w[i+1]
            if i==b[-1]:a=np.linspace(0,1,points/len(self.w)) # include the 1 in the last iteration
            for j in a:
                w=w1*j+(1-j)*w0
                weights.append(np.copy(w))
                mu.append(np.dot(w.T,self.mean)[0,0])
                sigma.append(np.dot(np.dot(w.T,self.covar),w)[0,0]**.5)
        return mu,sigma,weights
#---------------------------------------------------------------
#---------------------------------------------------------------

def get_history(symbols, start, end, data_path):
    
    """ to get Yahoo data from saved csv files. If the file does not exist for the symbol, 
    data is read from Yahoo finance and the csv saved.
    symbols: symbol list
    start, end : datetime start/end dates
    data_path : datapath for csv files - use double \\ and terminate path with \\
    """  

    symbols_ls = list(symbols)
    for ticker in symbols:
        print ticker,
        try:
            #see if csv data available
            data = pd.read_csv(data_path + ticker + '.csv', index_col='Date', parse_dates=True)
        except:
            #if no csv data, create an empty dataframe
            data = pd.DataFrame(data=None, index=[start])

        #check if there is data for the start-end data range

        if start.toordinal() < data.index[0].toordinal() \
                             or end.toordinal() > data.index[-1].toordinal():

            print 'Refresh data.. ',
            try:
                new_data = web.get_data_yahoo(ticker, start, end)

                if new_data.empty==False:
                    if data.empty==False:
                        try:
                            ticker_data = data.append(new_data).groupby(level=0, by=['rownum']).last()
                        except:
                            print 'Merge failed.. '
                    else:
                        ticker_data = new_data
                    try:
                        ticker_data.to_csv(data_path + ticker + '.csv')
                        print ' UPDATED.. '
                    except:
                        print 'Save failed.. '
                else:
                    print 'No new data.. '
            except:
                print 'Download failed.. '
                # remove symbol from list
                symbols_ls.remove(ticker)
        else:
            print 'OK.. '
        pass

    pdata = pd.Panel(dict((symbols_ls[i], pd.read_csv(data_path + symbols_ls[i] + '.csv',\
                     index_col='Date', parse_dates=True).sort(ascending=True)) for i in range(len(symbols_ls))) )


    return pdata.ix[:, start:end, :]

def get_trading_dates(start, end, offset=0):
    
    ''' to create a list of trading dates (timestamps) for use with Zipline or Quantopian.
         offset = 0 -> 1st trading day of month, offset = -1 -> last trading day of month.
         start, end are datetime.dates'''

    trading_dates = list([])

    trading_days= tradingcalendar.get_trading_days(start, end)

    month = trading_days[0].month
    for i in range(len(trading_days)) :
        if trading_days[i].month != month :
            try :
                trading_dates = trading_dates + list([trading_days[i + offset]])
            except :
                raise

            month = trading_days[i].month

    return trading_dates