import pandas as pd
import numpy as np
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


def mAvg_trading_signals(ticker, period='3mo', interval='1d'):
    data = yf.download(ticker, period = '3mo', interval="1d")['Adj Close']
    df = pd.DataFrame(data)
    df.fillna(0, inplace = True)
    df_daily_close = df
    df_daily_close.fillna(0, inplace = True)

    df_daily_pct_c = df_daily_close.pct_change()
    df_daily_pct_c.fillna(0, inplace = True)
    df_daily_pct_c = df_daily_close / df_daily_close.shift(1) - 1
    df_daily_pct_c.fillna(0, inplace = True)
    df['Daily_S_RoR'] = df_daily_pct_c * 100  

    # LOG Rate Of Return
    df_daily_log_returns = np.log(df_daily_close.pct_change() + 1)
    df_daily_log_returns.head()
    df_daily_log_returns.fillna(0, inplace = True)
    df['Daily_Log'] = df_daily_log_returns['Adj Close'] * 100 

    # Total Return
    df_cum_daily_return = (1 + df_daily_pct_c).cumprod()
    df['Total_RoR'] = df_cum_daily_return
    df.rename(columns={'Adj Close': ticker}, inplace=True)

    # Build MovingAverages
    short_window = 5
    long_window = 22
    period = 20
    multiplier = 2
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0.0
    signals['short_mavg'] = df[ticker].rolling(window=short_window,min_periods=1,center=False).mean()
    signals['long_mavg'] = df[ticker].rolling(window=long_window, min_periods=1, center=False).mean()
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:]> signals['long_mavg'][short_window:],1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    signals['UpperBand'] = df[ticker].rolling(period).mean() + df[ticker].rolling(period).std() * multiplier
    signals['LowerBand'] = df[ticker].rolling(period).mean() - df[ticker].rolling(period).std() * multiplier

    fig = plt.figure(figsize=(18, 7), dpi = 100)

    ax1 = fig.add_subplot(111,  ylabel='Price in $')
    df[ticker].plot(ax=ax1, lw=3., color = 'k')
    ax1.plot(signals.short_mavg, '--', color='c', lw=2)
    ax1.plot(signals.long_mavg, '--', color='m', lw=2)
    ax1.plot(signals.UpperBand, '-', color = 'y', lw=2)
    ax1.plot(signals.LowerBand, '-', color = 'y', lw=2)
    ax1.plot(signals.loc[signals.positions == 1.0].index, signals.short_mavg[signals.positions == 1.0],'^', markersize=20, color = 'g')
    ax1.plot(signals.loc[signals.positions == -1.0].index, signals.short_mavg[signals.positions == -1.0],'v', markersize=20, color = 'r')
    ax1.vlines(signals.short_mavg[signals.positions == 1.0].index, df[ticker].min()-5, df[ticker].max()+5, linestyles ="solid", colors ="g", lw=3)
    ax1.vlines(signals.short_mavg[signals.positions == -1.0].index, df[ticker].min()-5, df[ticker].max()+5, linestyles ="solid", colors ="r", lw=3)
    ax1.set_title(ticker+' - Moving Average Trade Signals SHORT-TERM', fontsize=25)
    ax1.legend(signals[['short_mavg','long_mavg','UpperBand','LowerBand']])
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    plt.show()

    print('Time To Buy:\n  ', signals.short_mavg[signals.positions == 1.0])
    print('\n\nTime To Sell: \n', signals.short_mavg[signals.positions == -1.0],'\n')
    return signals

def gainers():
    gainers = (get_day_gainers())
    print(gainers.Symbol[:5])
    for g in gainers.Symbol[:5]:
        signal = mAvg_trading_signals(ticker = g)
        print(signal.tail(),'\n\n')

def losers():
    losers = (get_day_losers())
    print(losers.Symbol[:5])
    for l in losers.Symbol[:5]:
        signal = mAvg_trading_signals(ticker = l)
        print(signal.tail(),'\n\n')            


# if __name__ == "__main__":    
    # signals = mAvg_trading_signals(ticker = 'AAPL') 
    # gainers()
    # losers()