# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:46:25 2018

@author: rl17174
"""

import os
os.chdir(r'C:\Users\bs40027\Desktop\qsproject\bin')

from strategy import CorpusStrategy
from data  import OHLCData, CommodityFutureData
from portfolio import EqualWeightPort
from backtest import Backtest
from matplotlib.backends.backend_pdf import PdfPages

###############################################################################
#********************************* spot data **********************************
n = 6
# process data
SHcomp = OHLCData()
data_dir = '../data/OHLC data/000001.SS.csv'
SHcomp.read(data_dir)
price_ts = SHcomp.get_clean_price_ts('Adj Close')
train_ts = price_ts['1995':'2004']
test_ts = price_ts['2005':'2013']
# create strategy
strat = CorpusStrategy(train_ts, n)
strat.get_vectorized_signal(test_ts)
# accept signal and produce position
port = EqualWeightPort()
port.get_position(strat.signals)
# backtest
my_bt = Backtest(test_ts[port.positions.index], port.positions)
stats = my_bt.output_summary_stats(rf=0.02)
###############################################################################



###############################################################################
#******************************** future data *********************************
# preprocess data
data_dir = '../data/future data'
day_dir = 'day'
night_dir = 'night'
concat_dir = os.path.join(data_dir, 'concat')
if not os.path.exists(concat_dir):
    os.makedirs(concat_dir)
main_contracts = CommodityFutureData.get_main_contracts(os.path.join(data_dir, 
                                                                     day_dir))
CommodityFutureData.concat_data(main_contracts, concat_dir, data_dir, day_dir, 
                                night_dir)
# process data
n = 6
trading_time_slots = ('09:00:00', '10:15:00', '10:30:00', '11:30:00', 
                      '13:30:00', '15:00:00', '21:00:00', '02:30:00')
outliers = (-1, 0)
ag = CommodityFutureData(trading_time_slots, outliers)

output_dir = '../output'
multi_pdf = PdfPages(os.path.join(output_dir, 'equity_curve.pdf'))
res = []
files = os.listdir(concat_dir)
for file in files:
    concat_data_dir = os.path.join(concat_dir, file)
    ag.read(concat_data_dir)
    price_ts = ag.get_clean_price_ts('MidPrice')
    train_ts = price_ts.iloc[:3000]
    test_ts = price_ts.iloc[3000:]
    # create strategy
    strat = CorpusStrategy(train_ts, n)
    strat.get_vectorized_signal(test_ts)
    # accept signal and produce position
    port = EqualWeightPort()
    port.get_position(strat.signals)
    # backtest
    my_bt = Backtest(test_ts[port.positions.index], port.positions)
    stats = my_bt.output_summary_stats(multi_pdf=multi_pdf, periods=1)
    res.append((file.split('_')[1], stats))
multi_pdf.close()
###############################################################################
