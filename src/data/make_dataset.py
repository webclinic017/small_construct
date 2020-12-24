import pandas as pd
import yfinance as yf
from yahoo_fin.stock_info import tickers_dow, tickers_nasdaq, tickers_other, tickers_sp500
from sklearn.externals import joblib

path = '/home/gordon/work/small_builds/small_construct/data/raw/'
path_stockTickers = '/home/gordon/work/small_builds/small_construct/data/raw/stock_tickers/'
path_historicalPrices = '/home/gordon/work/small_builds/small_construct/data/raw/stock_history/'
# dow, sp500, nasdaq, other = tickers_dow(), tickers_sp500(), tickers_nasdaq(), tickers_other()

class StockMarket(object):
    def __init__(self):
        # self.dow, self.sp500, self.nasdaq, self.other = dow, sp500, nasdaq, other
        # self.ticker_lists = [self.dow, self.sp500, self.nasdaq, self.other]
        self.saveName_lst = ['dowTickers_','sp500tickers_','nasdaqTickers_','otherTickers_']            

        # for t in range(len(self.ticker_lists)):
        #     self.new_path = path_stockTickers + self.saveName_lst[t] + '.pkl'
        #     joblib.dump(self.ticker_lists[t], self.new_path)


    def get_stock_price_history(self, ticker_set, saveName, period = '2y', interval = '1d'):
        self.ticker_set, self.saveName = ticker_set, saveName
        self.period, self.interval = period, interval

        self.df = yf.download(self.ticker_set, period = self.period, interval = self.interval)['Adj Close']
        self.df.to_csv(path_historicalPrices + self.saveName + '_' + self.period + '_'+ self.interval + '.csv')
        joblib.dump(self.df, path_historicalPrices+ self.saveName + '_' + self.period + '_'+ self.interval +'.pkl')


if __name__ == '__main__':
    # x = StockMarket()
    # x.stockHistoryImport()
    pass