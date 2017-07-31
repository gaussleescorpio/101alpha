import numpy as np
import pandas as pd
from numpy import abs
from numpy import log
from numpy import sign
from scipy.stats import rankdata

def ts_sum(df,window=10):
    return df.rolling(window).sum()

def sma(df,window=10):
    return df.rolling(window).mean()

def stddev(df,window=10):
    return df.rolling(window).std()

def correlation(x,y,window=10):
    return x.rolling(window).corr(y)

def covariance(x,y,window=10):
    return x.rolling(window).cov(y)

def rolling_rank(na):
    '''
    auxiliary function to be used in pd.rolling_apply
    na is a numpy array
    return the rank of the last value in the array
    '''
    return np.argsort(np.argsort(na))[-1]

def ts_rank(df,window=10):
    return df.rolling(window).apply(rolling_rank)

def rolling_prod(na):
    return np.prod(na)

def product(df,window=10):
    return df.rolling(window).apply(rolling_prod)

def ts_min(df,window=10):
    return df.rolling(window).min()

def ts_max(df,window=10):
    return df.rolling(window).max()

def delta(df,period=1):
    return df.diff(period)

def delay(df,period=1):
    return df.shift(period)

def rank(df):
    '''
    cross sectional rank
    return a df with rank along columns
    '''
    return df.rank(axis=1,pct=True)

def scale(df,k=1):
    '''
    scaling time series
    a df rescaled such that sum(abs(df))=k
    '''
    return df.mul(k).div(np.abs(df).sum(axis=1))

def ts_argmax(df,window=10):
    return df.rolling(window).apply(np.argmax)+1

def ts_argmin(df,window=10):
    return df.rolling(window).apply(np.argmin)+1

def decay_linear(df,period=10):
    
    if df.isnull().values.any():
        df.fillna(method="ffill",inplace=True)
        df.fillna(method="bfill",inplace=True)
        df.fillna(value=0,inplace=True)
    
    na_lwma=np.zeros_like(df)
    na_lwma[:period,:]=df.ix[:period,:]
    na_series=df.as_matrix()
    
    divisor=period*(period+1)/2
    y=(np.arange(period)+1)*1.0/divisor
    
    for row in range(period-1,df.shape[0]):
        x=na_series[row-period+1:row+1,:]
        na_lwma[row,:]=(np.dot(x.T,y))
    return pd.DataFrame(na_lwma,index=df.index,columns=df.columns)


    
    
    
    
    


    


















    






