from src.data.make_dataset import StockMarket
from sklearn.externals import joblib
import pandas as pd


if __name__ == '__main__':
    x = StockMarket()

# GET_TICKERS-SAVE_TICKERS-IMPORT_TICKERS---GET_DATA-SAVE_DATA-IMPORT_DATA
    # path = '/home/gordon/work/small_builds/small_construct/data/raw/stock_tickers/dowTickers_.pkl'
    # path = '/home/gordon/work/small_builds/small_construct/data/raw/stock_tickers/nasdaqTickers_.pkl
    # path = '/home/gordon/work/small_builds/small_construct/data/raw/stock_tickers/otherTickers_.pkl'
    # path = '/home/gordon/work/small_builds/small_construct/data/raw/stock_tickers/sp500tickers_.pkl'
    sp500tickers = open('data/raw/stock_tickers/sp500tickers_.pkl', 'rb')
    ticker_set = joblib.load(sp500tickers)
    # x.get_stock_price_history(ticker_set, 'sp500tickers_priceHistory_')

# MOVING_AVERAGES:
    ticker = 'AAPL'
    # signals = mAvg_trading_signals() 
    # gainers()
    # losers()
