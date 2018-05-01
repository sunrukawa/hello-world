# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:46:25 2018

@author: rl17174
"""

import os
os.chdir('/Users/sunrukawa/Desktop/python/qishi_qr/bin/')

from strategy import CorpusStrategy
from data import OHLCData
from portfolio import EqualWeightPort
from backtest import Backtest

n = 6
# process data
OHLC = OHLCData()
data_dir = '../data/OHLC data/000001.SS.csv'
OHLC.read(data_dir)
price_ts = OHLC.get_clean_price_ts('Adj Close')
train_ts = price_ts['1995':'2004']
test_ts = price_ts['2005':'2013']
# create strategy
strat = CorpusStrategy(train_ts, n)
strat.get_vectorized_signal(test_ts)
# accept signal and produce position
port = EqualWeightPort()
port.get_position(strat.signals.iloc[(n-1):])
# backtest
my_bt = Backtest(test_ts[port.positions.index], port.positions)
stats = my_bt.output_summary_stats(rf=0.02)
