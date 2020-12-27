from src.data.make_dataset import StockMarket
from src.models.movingAverage import MovingAverageGenerator
import joblib
# from sklearn.externals import joblib
import pandas as pd


if __name__ == '__main__':
# make_dataset.py ~ STOCK_PRICE_HISTORY - (single_ticker, multiple_tickers, index_components)
    priceHistory = StockMarket()

    pull_price_history = input('Would you like to pull some stock price history?')
    if pull_price_history.lower() == 'yes':
        single = input('Pull Price History For - Single Ticker?')
        if single.lower() == 'yes':
            single_ticker = input('Please Enter Your Single Ticker:')
            priceHistory.get_single_stock_price_history(single_ticker)

        multiple_tickers = input('Please Enter Your Stocks (seperate with space):')
        if multiple_tickers.lower() == 'yes':
            priceHistory.get_multiple_stock_price_history(multiple_tickers, 'mini_portfolio')

        index_tickers = input('Would you like to pull history for an index(dow, sp500, nasdaq): ')
        if index_tickers == 'dow':
            priceHistory.get_indexComponents_price_history(dowTickers_)
        elif index_tickers == 'sp500':
            priceHistory.get_indexComponents_price_history(sp500tickers_)
        elif index_tickers == 'nasdaq':
            priceHistory.get_indexComponents_price_history(nasdaqTickers_)
    # saveName = 'sp500tickers_' # 'dowTickers_', 'nasdaqTickers_', 'otherTickers_'
    


# movingAverage.py ~ MOVING_AVERAGES:
    mavg = MovingAverageGenerator()

    tic = input('Hello, Is there a specific Stock you would like to generate a Moving Average For?')
    if tic.lower() == 'yes':
        ticker = input('Please Enter The Ticker:')
        mavg.mAvg_trading_signals(ticker) 
    
    gain = input('Would you like to generate Moving Averages for the TOP-5-STOCK-GAINERS from today? ')
    if gain.lower() == 'yes':
        mavg.gainers()
    
    lose = input('Would you like to generate Moving Averages for the TOP-5-STOCK-LOSERS from today? ')
    if lose.lower() == 'yes':
        mavg.losers()
    
    print('\n   Thank you, have a great day & happy trading!\n')