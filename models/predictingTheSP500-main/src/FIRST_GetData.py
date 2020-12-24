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
import requests
import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc
from scipy import stats
from datetime import *
today = date.today()
print('\n          * * * NO ISSUES - ALL IMPORTS LOADED * * * \n')

class Insource:
    def __init__(self, sp_ticker, index_tickers, interval):
        self.sp_ticker = sp_ticker
        self.index_tickers = index_tickers
        self.interval = interval

    def scrape(self):
        self.table = pd.read_html('https://bigcharts.marketwatch.com/markets/indexes.asp')
        self.table_df = self.table[0]
        self.table_df.to_csv('data/tickers/S&P500-Info.csv')

        self.index_name = []
        for n in range(len(self.table_df)):
            lst = [0,4,5,6,13]
            if n in lst:
                self.index_name.append(self.table_df.iloc[n][0])
        self.index_names = []
        for i in self.index_name:
            self.lst = list(i)
            self.columns = self.lst
            self.col = [column.replace(' ', '_') for column in self.columns]
            self.string = ''
            for c in self.col:
                self.string += c
            self.index_names.append(self.string)
        return self.index_names

    def IndexHistory(self):
        self.scrape()
        for n in range(len(self.interval)):
            self.df0 = yf.download(self.index_tickers, period='10y', interval=self.interval[n], parse_dates=True, index_col=0)['Adj Close']
            self.df0.columns = ['SP500','DOW','NASDAQ','RUSSELL2000']
            self.df0.to_csv('data/history/' + 'SP500_DOW_NASDAQ_RUSSELL2000_'+self.interval[n]+'.csv')
            
            self.df1 = yf.download(self.sp_ticker, period='10y', interval=self.interval[n], parse_dates=True, index_col=0)
            self.df1.to_csv('data/history/' + self.index_names[3] + self.interval[n]+'.csv')

if __name__ == '__main__':
    sp_ticker, index_tickers, interval = '^GSPC', ['^GSPC','^DJI','^IXIC','^RUT'], ['1d', '1wk', '1mo']
    x = Insource (sp_ticker, index_tickers, interval)
    x.IndexHistory()