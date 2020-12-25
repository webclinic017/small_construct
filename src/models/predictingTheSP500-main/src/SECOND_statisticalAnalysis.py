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

class Analysis:
    def __init__(self, stock, stocks, dataname1, dataname2):
        self.stock = stock
        self.stocks = stocks
        self.dataname1 = dataname1
        self.dataname2 = dataname2

    def GetData(self):
        self.df1 = pd.read_csv('data/history/' + self.dataname1 + '.csv')
        self.df2 = pd.read_csv('data/history/' + self.dataname2 + '.csv')
        return self.df1, self.df2

    def df_top_bottom_plot(self):
        self.data_df = yf.download('^GSPC', period='max', interval='1d')
        self.prices = self.data_df['Adj Close']
        self.volumes = self.data_df['Volume']
    # The top plot consisting of dailing closing prices
        self.top = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=4)
        self.top.plot(self.prices.index, self.prices, label='Adj Close')
        plt.title(f'{self.stock} Last Price from 2015 - 2018')
        plt.legend(loc=2)
    # The bottom plot consisting of daily trading volume
        self.bottom = plt.subplot2grid((4, 4), (3,0), rowspan=1, colspan=4)
        self.bottom.bar(self.volumes.index, self.volumes)
        plt.title(f'{self.stock} Daily Trading Volume')
        plt.gcf().set_size_inches(12, 8)
        plt.subplots_adjust(hspace=0.75)
        plt.show()

    def create_candlestick(self, start='2020-08-01', end='2020-11-01'):
        self.start=start
        self.end=end
        self.df_subset = yf.download('^GSPC', start=self.start, end=self.end, interval='1d')
        self.df_subset['Date'] = self.df_subset.index.map(mdates.date2num)
        self.df_ohlc = self.df_subset[['Date','Open', 'High', 'Low', 'Close', 'Volume']]

        figure, ax = plt.subplots(figsize = (12,8), dpi = 100)
        formatter = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(formatter)
        candlestick_ohlc(ax, self.df_ohlc.values, width=0.8, colorup='green', colordown='red')
        plt.show()

        self.df_ohlc['Date'] = pd.to_datetime(self.df_ohlc['Date'])
        self.df_ohlc.set_index('Date', inplace = True)
        mpf.plot(self.df_ohlc, type='candle', mav=(20, 50, 200), volume=True, width_adjuster_version='v0', figsize=(12,8))
        plt.show()

    def plot_returns(self, fd):
        self.fd = fd
        self.daily_changes = self.fd['Adj Close'].pct_change(periods = 1)
        plt.title(f'Daily Returns {self.stock}')
        self.daily_changes.plot()
        plt.show()
        
        self.cumulative_returns = self.daily_changes.cumsum()
        self.cumulative_returns.plot()
        plt.title('Cumulative Annual Returns')
        plt.show()
        
        self.daily_changes.hist(bins=50, figsize=(8, 4))
        plt.title('Histogram Plot - Annual Returns')
        plt.show()

        self.df_vol = self.fd['Adj Close'].pct_change()
        self.df_std = self.df_vol.rolling(window=30, min_periods=30).std()
        self.df_std.plot()
        plt.title('Volatility Plot - Annual Returns')
        plt.show()
        
        self.daily_move = self.fd['Adj Close'].pct_change(periods = 1).dropna()
        figure = plt.figure(figsize = (8,4))
        ax = figure.add_subplot(111)
        stats.probplot(self.daily_move, dist = 'norm', plot = ax)
        plt.show()

    def mavg(self, data):
        self.data = data
        self.data.set_index('Date', inplace = True)
        self.df_last = self.data['Adj Close']
        self.series_short = self.df_last.rolling(window=20, min_periods=20).mean()
        self.series_long = self.df_last.rolling(window=50, min_periods=50).mean()
        self.df_sma = pd.DataFrame(columns=['short', 'long'])
        self.df_sma['short'] = self.series_short
        self.df_sma['long'] = self.series_long
        self.df_sma.plot(figsize=(9, 6))
        plt.title(f'SIMPLE MOVING AVERAGE - {stock}')
        plt.show()

        self.series_shortE = self.df_last.ewm(span=5).mean()
        self.series_longE = self.df_last.ewm(span=30).mean()
        self.df_smaE = pd.DataFrame(columns=['short', 'long'])
        self.df_smaE['short'] = self.series_shortE
        self.df_smaE['long'] = self.series_longE
        self.df_smaE.plot(figsize=(9, 6))
        plt.title(f'EXPONENTIAL MOVING AVERAGE - {stock}')
        plt.show()
        

if __name__ == '__main__':
    stock, stocks = 'SP500', ['SP500','DOW','NASDAQ','RUSSELL2000']
    dataname1, dataname2 = 'S&P_500_Index1d', 'SP500_DOW_NASDAQ_RUSSELL2000_1d'

    x = Analysis(stock, stocks, dataname1, dataname2)

    df_sp, df_index = x.GetData()
    x.df_top_bottom_plot()
    x.create_candlestick()
    x.plot_returns(df_sp)
    x.mavg(df_sp)