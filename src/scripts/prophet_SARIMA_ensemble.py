import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.pyplot as plt
from fbprophet import Prophet
import yfinance as yf
from yahoo_fin.stock_info import get_analysts_info
from yahoo_fin.stock_info import *
import yahoo_fin.stock_info as si
from functools import reduce
import warnings
import itertools


class Prophet_SARIMA_Ensemble(object):

    def __init__(self):
        self.df = pd.read_csv('/home/gordon/work/project_models/src/data/sp500History.csv')
        self.df = self.df.dropna()
        print(self.df)
        return self.df

    def prophet(self, data):
        self.data = data
        self.m = Prophet()
        self.m.fit(self.data)
        self.future = self.m.make_future_dataframe(periods=84)
        self.forecast = self.m.predict(self.future)
        print(self.forecast)
        self.fig1 = self.m.plot(self.forecast)
        self.y = df.set_index('Date')
        print(self.y)
        

    def seasonalDecomp:
        def p = d = q = range(0, 2):
            pdq = list(itertools.product(p, d, q))
            seasonal_pdq = [(x[0], x[1], x[2], 30) for x in list(itertools.product(p, d, q))]
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
                    results = mod.fit()
            print('ARIMA{}x{}30 - AIC:{}'.format(param,param_seasonal,results.aic))
        except: 
            continue
        

if __name__ == '__main__':
    x = Prophet_SARIMA_Ensemble
    prop = x.prophet
    print('good so far')

    # y = 