import pandas as pd
import numpy as np
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
plt.style.use('ggplot')
matplotlib.rcParams.update({'font.family' : 'sans'})
sm, med, lg = 10, 15, 20
plt.rc('font', size = sm)         # controls default text sizes
plt.rc('axes', titlesize = med)   # fontsize of the axes title
plt.rc('axes', labelsize = med)   # fontsize of the x & y labels
plt.rc('xtick', labelsize = sm)   # fontsize of the tick labels
plt.rc('ytick', labelsize = sm)   # fontsize of the tick labels
plt.rc('legend', fontsize = sm)   # legend fontsize
plt.rc('figure', titlesize = lg)  # fontsize of the figure title
plt.rc('axes', linewidth=2)       # linewidth of plot lines
import yfinance as yf
import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc
from scipy import stats
from datetime import *
today = date.today()
print('\n          * * * NO ISSUES - ALL IMPORTS LOADED * * * \n')

class Segment2:
    def __init__(self, stocks, period, interval, dataname, column_name, parse_dates = True):
        self.stocks = stocks
        self.period = period
        self.interval = interval
        self.parse_dates = parse_dates
        self.dataname = dataname
        self.column_name = column_name

    def IndexHistory(self):
        self.df = yf.download(
            self.stocks, period = self.period, interval = self.interval, 
            parse_dates = self.parse_dates, index_col=0)['Adj Close'
            ]
        print('Historical Data For All Index Pulled Successfully')
        self.SaveData()
        self.GetData()
        return self.df1

    def SaveData(self):
        self.df.to_csv('data/ARIMA/' + self.dataname + '.csv')
        print('Data Saved To csv Successfully')

    def GetData(self):
        self.df1 = pd.read_csv('data/ARIMA/' + self.dataname + '.csv', index_col='Date')
        self.df1.columns = self.column_name
        self.df1 = self.df1.fillna(0)
        self.df1.index = pd.to_datetime(self.df1.index)
        print('Historical df Has been Imported Successfully, and is ready to be used.')
        return self.df1        

    def correlation(self):
        self.GetData()
        return self.df1.pct_change().corr()

    # def covariance()




if __name__ == '__main__':
    stocks, dataname = ['^GSPC', '^IXIC', '^DJI', 'RTY=F'], 'sp500_nasdaq_dow_russell_10y_1d'
    column_name, period, interval = ['Russell2000', 'DOW', 'SP500','NASDAQ'], '10y','1d'
    x = Segment2(stocks, period, interval, dataname, column_name)
    # stocks_data = x.IndexHistory()
    stocks_data = x.GetData()
    print(stocks_data.head())
    print(len(stocks_data))

    print(x.correlation())

