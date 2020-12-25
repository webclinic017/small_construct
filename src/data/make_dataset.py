import pandas as pd
import yfinance as yf
from yahoo_fin.stock_info import tickers_dow, tickers_nasdaq, tickers_other, tickers_sp500
from sklearn.externals import joblib


class StockMarket(object):
    def __init__(self):
        self.path = '/home/gordon/work/small_construct/data/raw/'
        self.path_stockTickers = '/home/gordon/work/small_construct/data/raw/stock_tickers/'
        self.path_historicalPrices = '/home/gordon/work/small_construct/data/raw/stock_history/'
        self.saveName_lst = ['dowTickers_','sp500tickers_','nasdaqTickers_','otherTickers_']
        # dow, sp500, nasdaq, other = tickers_dow(), tickers_sp500(), tickers_nasdaq(), tickers_other()
        # self.dow, self.sp500, self.nasdaq, self.other = dow, sp500, nasdaq, other
        # self.ticker_lists = [self.dow, self.sp500, self.nasdaq, self.other] 
        # for t in range(len(self.ticker_lists)):
        #     self.new_path = path_stockTickers + self.saveName_lst[t] + '.pkl'
        #     joblib.dump(self.ticker_lists[t], self.new_path)

    def get_single_stock_price_history(self, ticker, period = '2y', interval = '1d'):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.df = yf.download(self.ticker, period=self.period, interval=self.interval)['Adj Close']
        self.df.to_csv(self.path_historicalPrices + self.ticker + '_' + self.period + '_' + self.interval + '.csv')
        joblib.dump(self.df, self.path_historicalPrices + self.ticker + '_' + self.period + '_'+ self.interval +'.pkl')

    def get_multiple_stock_price_history(self, ticker_set, saveName, period = '2y', interval = '1d'):
        self.ticker_set = ticker_set
        self.saveName = saveName
        self.period = period
        self.interval = interval        
        self.df = yf.download(self.ticker_set, period=self.period, interval=self.interval)['Adj Close']
        self.df.to_csv(self.path_historicalPrices + self.saveName + '_' + self.period + '_'+ self.interval + '.csv')
        joblib.dump(self.df, self.path_historicalPrices + self.saveName + '_' + self.period + '_' + self.interval +'.pkl')

    def get_indexComponents_price_history(self, saveName, period = '2y', interval = '1d'):
        self.saveName = saveName
        self.period = period
        self.interval = interval
        self.new_path = (self.path_stockTickers + self.saveName + '.pkl')
        self.ticker_symbols = open(self.new_path, 'rb')
        self.ticker_set = joblib.load(self.ticker_symbols)
        self.df = yf.download(self.ticker_set, period=self.period, interval=self.interval)['Adj Close']
        self.df.to_csv(self.path_historicalPrices + self.saveName + '_' + self.period + '_'+ self.interval + '.csv')
        joblib.dump(self.df, self.path_historicalPrices + self.saveName + '_' + self.period + '_' + self.interval +'.pkl')             


if __name__ == '__main__':
    priceHistory = StockMarket()

    # priceHistory.get_single_stock_price_history('TSLA')
    
    # priceHistory.get_multiple_stock_price_history(['AAPL','TSLA','AMZN','NFLX'], 'mini_portfolio')

    # saveName = 'sp500tickers_' # 'dowTickers_', 'nasdaqTickers_', 'otherTickers_'
    # priceHistory.get_indexComponents_price_history(saveName)