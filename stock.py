import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datetime import date
import itertools
import math
import yfinance as yf
from typing import List

class PortfolioBuilder:

    def get_daily_data(self, tickers_list: List[str], start_date: date, end_date: date = date.today()) -> pd.DataFrame:
        """
        get stock tickers adj_close price for specified dates.

        :param List[str] tickers_list: stock tickers names as a list of strings.
        :param date start_date: first date for query
        :param date end_date: optional, last date for query, if not used assumes today
        :return: daily adjusted close price data as a pandas DataFrame
        :rtype: pd.DataFrame

        example call: get_daily_data(['GOOG', 'INTC', 'MSFT', ''AAPL'], date(2018, 12, 31), date(2019, 12, 31))
        """
        try:
            data = web.DataReader(tickers_list, "yahoo", start_date, end_date)["Adj Close"]
            self.tickers_list = tickers_list
            self.data_array = pd.DataFrame(data).to_numpy()
            self.row_data = np.size(self.data_array,0)
            if data.isnull().values.any():
                raise ValueError
        except:
            ValueError
        return data

    def rec_permutation(self,data, perm_length):
        """
        This function is a recursive implementation of choosing "k" elements
        out of "n" options with returning occurrences
        :param data: the "n" length data base
        :param perm_length: the number of elements to choose "k"
        :return: res_list: a list of lists of all possible permutations
        """
        if perm_length == 1:
            return [[atom] for atom in data]

        res_list = []  # how can we make it better ?
        smaller_perm = self.rec_permutation(data, perm_length - 1)
        for elem in data:
            for sub_combination in smaller_perm:
                res_list.append([elem] + sub_combination)
        return res_list

    def xt_builder(self, data):
        xt_list = []
        for i in range(1, np.size(data,0)):
            xt_list.append((np.divide(data[i], data[i-1])).tolist())
        xt = np.array(xt_list)
        return xt

    def bw_builder(self,a,tickers_list):       #return numpy.ndarray
        x = (1/a)*100
        uf_bw = PortfolioBuilder().rec_permutation(np.arange(0,100+x,x),len(tickers_list))
        bw_100 = []
        for i in uf_bw:
            if sum(i) == 100:
                bw_100.append(i)
        bw = np.divide(bw_100,100)
        return bw

    def sbw_builder(self,bw,xt):
        sbw_lst = []
        for i in range(0, len(bw)):
            sbwi = []
            bwi = bw[i]
            for j in range(0, len(xt)):
                xj = xt[j]
                if j == 0:
                    sbwi.append((np.dot(bwi, xj)))
                else:
                    sbwi.append(((np.dot(bwi, xj))*sbwi[-1]))
            sbw_lst.append(sbwi)
        sbw = np.array(sbw_lst)
        return sbw

    def uni_bt_builder(self,sbw,bw):
        #up_bt
        sbw_dot_bw_list = []
        for i in range(0, len(bw)):
            b = []
            bwi = bw[i]
            for j in range(0, np.size(sbw, axis=1)):
                sbwij = sbw[i, j]
                b.append((bwi * sbwij).tolist())
            sbw_dot_bw_list.append(b)
        sbw_dot_bw = np.array(sbw_dot_bw_list)
        up_bt = np.sum(sbw_dot_bw, axis=0)
        # down_bt
        down_bt = np.sum(sbw, axis=0)
        # bt
        b0 = [(1 / len(self.tickers_list)) for k in range(0,len(self.tickers_list))]
        bt_list = [b0] #[b0]
        for l in range(0, np.size(sbw, axis=1)):
            bti = np.divide(up_bt[l], down_bt[l]).tolist()
            bt_list.append(bti)
        bt = np.array(bt_list)
        return bt

    def sbt_builder(self,xt,bt):
        sbt = [1.]
        for i in range(np.size(xt, axis=0)):
            xi = xt[i]
            bti = bt[i]
            if i == 0:
                sbt.append((np.dot(bti, xi)))
            else:
                sbt.append(((np.dot(bti, xi))*sbt[-1]))
        return sbt

    def find_universal_portfolio(self, portfolio_quantization: int = 20) -> List[float]:
        xt = self.xt_builder(self.data_array)
        bw = self.bw_builder(portfolio_quantization,self.tickers_list)
        sbw = self.sbw_builder(bw,xt)
        bt = self.uni_bt_builder(sbw,bw)
        sbt = self.sbt_builder(xt,bt)
        return sbt

    def exp_builder(self,n,xt,bt,t,i):
        up_exp = n*xt[t,i]
        bt_t = bt[t]
        xt_t = xt[t]
        down_exp = np.dot(bt_t, xt_t)
        x = np.divide(up_exp,down_exp)
        e = np.exp(x)
        return e

    def exponential_bt_builder(self,xt,n,row_data):
        b0 = [(1 / len(self.tickers_list)) for k in range(0,len(self.tickers_list))]
        bt_list = [b0]
        for t in range(0, row_data-1):
            bt = np.array(bt_list)
            btj = []
            for j in range(0, len(self.tickers_list)):
                expj = self.exp_builder(n,xt,bt,t,j)
                up_bt_j = np.multiply(bt[t,j],expj)
                down_bt_to_sum = []
                for k in range(0, len(self.tickers_list)):
                    btk = bt[t,k]
                    expk = self.exp_builder(n,xt,bt,t,k)
                    down_bt_to_sum.append((np.multiply(expk,btk)).tolist())
                down_bt_j = np.sum(down_bt_to_sum)
                btj.append(np.divide(up_bt_j,down_bt_j))
            bt_list.append(btj)
        return bt

    def find_exponential_gradient_portfolio(self, learn_rate: float = 0.5) -> List[float]:
        n = learn_rate
        xt = self.xt_builder(self.data_array)
        bt = self.exponential_bt_builder(xt,n,self.row_data)
        sbt = self.sbt_builder(xt,bt)
        return sbt


if __name__ == '__main__':  # You should keep this line for our auto-grading code.
    import time
    t0 = time.time()
    pb = PortfolioBuilder()
    df = pb.get_daily_data(['GOOG', 'AAPL', 'MSFT'], date(2020, 1, 1), date(2020, 2, 1))
    print(df)
    universal = pb.find_universal_portfolio(4)
    print("sbt_uni")
    print(universal)
    t1 = time.time()
    expo_grad = pb.find_exponential_gradient_portfolio()
    print("expo_grad[:9]")
    print(expo_grad[:9])
    print("expo_grad[10:18]")
    print(expo_grad[10:18])
    print("expo_grad[19:]")
    print(expo_grad[19:])
    t2 = time.time()
    print("time taken uni" , t1-t0, "sec")
    print("time taken expo" , t2-t1, "sec")
