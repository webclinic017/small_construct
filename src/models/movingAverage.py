import pandas as pd
import numpy as np
pd.options.display.max_rows    
pd.options.display.max_columns = 50
pd.options.display.max_rows = 999
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import cufflinks as cf
cf.go_offline(connected=True)
import yfinance as yf
from yahoo_fin.stock_info import *
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.family' : 'sans'})
sm, med, lg = 15, 20, 25
plt.rc('font', size = sm)         # controls default text sizes
plt.rc('axes', titlesize = med)   # fontsize of the axes title
plt.rc('axes', labelsize = med)   # fontsize of the x & y labels
plt.rc('xtick', labelsize = sm)   # fontsize of the tick labels
plt.rc('ytick', labelsize = sm)   # fontsize of the tick labels
plt.rc('legend', fontsize = sm)   # legend fontsize
plt.rc('figure', titlesize = lg)  # fontsize of the figure title
sns.set(style='darkgrid', context='talk', palette='Dark2')
import warnings
warnings.filterwarnings('ignore')


class MovingAverageGenerator(object):
    def __init__(self):
        pass

    def mAvg_trading_signals(self, ticker, period='3mo', interval='1d'):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.data = yf.download(self.ticker, period = self.period, interval = self.interval)['Adj Close']
        self.df = pd.DataFrame(self.data)
        self.df.fillna(0, inplace = True)
        self.df_daily_close = self.df
        self.df_daily_close.fillna(0, inplace = True)

        self.df_daily_pct_c = self.df_daily_close.pct_change()
        self.df_daily_pct_c.fillna(0, inplace = True)
        self.df_daily_pct_c = self.df_daily_close / self.df_daily_close.shift(1) - 1
        self.df_daily_pct_c.fillna(0, inplace = True)
        self.df['Daily_S_RoR'] = self.df_daily_pct_c * 100  

        # LOG Rate Of Return
        self.df_daily_log_returns = np.log(self.df_daily_close.pct_change() + 1)
        self.df_daily_log_returns.head()
        self.df_daily_log_returns.fillna(0, inplace = True)
        self.df['Daily_Log'] = self.df_daily_log_returns['Adj Close'] * 100 

        # Total Return
        self.df_cum_daily_return = (1 + self.df_daily_pct_c).cumprod()
        self.df['Total_RoR'] = self.df_cum_daily_return
        self.df.rename(columns={'Adj Close': self.ticker}, inplace=True)

        # Build MovingAverages
        self.short_window = 5
        self.long_window = 22
        self.period = 20
        self.multiplier = 2
        self.signals = pd.DataFrame(index=self.df.index)
        self.signals['signal'] = 0.0
        self.signals[self.ticker] = self.df[self.ticker]
        self.signals['short_mavg'] = self.df[self.ticker].rolling(
            window=self.short_window,min_periods=1,center=False).mean()
        self.signals['long_mavg'] = self.df[self.ticker].rolling(
            window=self.long_window, min_periods=1, center=False).mean()
        self.signals['signal'][self.short_window:] = np.where(
            self.signals['short_mavg'][self.short_window:]> self.signals['long_mavg'][self.short_window:],1.0, 0.0)
        self.signals['positions'] = self.signals['signal'].diff()
        self.signals['UpperBand'] = self.df[self.ticker].rolling(
            self.period).mean() + self.df[self.ticker].rolling(self.period).std() * self.multiplier
        self.signals['LowerBand'] = self.df[self.ticker].rolling(
            self.period).mean() - self.df[self.ticker].rolling(self.period).std() * self.multiplier

        fig = plt.figure(figsize=(18, 7), dpi = 100)

        ax1 = fig.add_subplot(111,  ylabel='Price in $')
        self.df[self.ticker].plot(ax=ax1, lw=3., color = 'k')
        # ax1.plot(self.df[self.ticker], lw=3, color='k')
        ax1.plot(self.signals.short_mavg, '--', color='c', lw=2)
        ax1.plot(self.signals.long_mavg, '--', color='m', lw=2)
        ax1.plot(self.signals.UpperBand, '-', color = 'b', lw=2)
        ax1.plot(self.signals.LowerBand, '-', color = 'b', lw=2)
        ax1.plot(
            self.signals.loc[self.signals.positions == 1.0].index,
            self.signals.short_mavg[self.signals.positions == 1.0]
            ,'^', markersize=20, color = 'g'
            )
        ax1.plot(
            self.signals.loc[self.signals.positions == -1.0].index, 
            self.signals.short_mavg[self.signals.positions == -1.0]
            ,'v', markersize=20, color = 'r'
            )
        ax1.vlines(
            self.signals.short_mavg[self.signals.positions == 1.0].index, 
            self.df[self.ticker].min(), self.df[self.ticker].max(), 
            linestyles ="solid", colors ="g", lw=2
            )
        ax1.vlines(
            self.signals.short_mavg[self.signals.positions == -1.0].index, 
            self.df[self.ticker].min(), self.df[self.ticker].max(), 
            linestyles ="solid", colors ="r", lw=2
            )
        ax1.set_title(self.ticker+' - Moving Average Trade Signals SHORT-TERM', fontsize=25)
        ax1.legend(self.signals[[ticker, 'short_mavg','long_mavg','UpperBand','LowerBand']])
        plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
        plt.show()

        print('Time To Buy:\n  ', self.signals.short_mavg[self.signals.positions == 1.0])
        print('\n\nTime To Sell: \n', self.signals.short_mavg[self.signals.positions == -1.0],'\n')
        return self.signals

    def gainers(self):
        self.gainers = (get_day_gainers())
        print(self.gainers.Symbol[:5])
        for g in self.gainers.Symbol[:5]:
            self.signal = self.mAvg_trading_signals(ticker = g)
            print(self.signal.tail(),'\n\n')

    def losers(self):
        self.losers = (get_day_losers())
        print(self.losers.Symbol[:5])
        for l in self.losers.Symbol[:5]:
            self.signal = self.mAvg_trading_signals(ticker = l)
            print(self.signal.tail(),'\n\n')            


if __name__ == "__main__":
    mavg = MovingAverageGenerator()
    mavg.mAvg_trading_signals(ticker = 'AAPL') 
    # mavg.gainers()
    # mavg.losers()