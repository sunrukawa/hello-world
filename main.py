# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:46:25 2018

@author: rl17174
"""

import os
os.chdir('/Users/sunrukawa/Desktop/python/qishi_qr/bin/')
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from strategy import CorpusStrategy, BuyHoldStrategy, RandomGuessStrategy
from data import CommodityFutureData
from portfolio import EqualWeightPort
from backtest import Backtest
from constants import trading_time_slots

###############################################################################
#************************************ NLP *************************************
# preprocess data
cft = 'zn'
future_data_dir = '../data/future data'
data_dir = os.path.join(future_data_dir, cft)
day_dir = os.path.join(data_dir, 'day')
night_dir = os.path.join(data_dir, 'night')
concat_dir = os.path.join(data_dir, 'subconcat')
if not os.path.exists(concat_dir):
    os.makedirs(concat_dir)
    main_contracts = CommodityFutureData.get_main_contracts(day_dir)
    mc_files = CommodityFutureData.get_avail_main_contracts_files(
                                                     main_contracts, night_dir)
    CommodityFutureData.concat_data(mc_files, concat_dir, day_dir, night_dir)
# process data
n = 6
outliers = (-1, 0)
cf = CommodityFutureData()
output_dir = '../output'
multi_ec = PdfPages(os.path.join(output_dir, 'equity_curve_corpus.pdf'))
multi_price = PdfPages(os.path.join(output_dir, 'price.pdf'))
res = []
files = os.listdir(concat_dir)
# specific for macOS
files = sorted(files)
if files[0].startswith('.'):
    files = files[1:]
#
for file in files:
    file_date = file.split('_')[1]
    concat_data_dir = os.path.join(concat_dir, file)
    cf.read(concat_data_dir)
    price_ts = cf.get_clean_price_ts('MidPrice', trading_time_slots[cft], 
                                     outliers)
    price_ts.plot(figsize=(20,15), fontsize=15)
    multi_price.savefig()
    plt.clf()
    split_time = pd.Timestamp(file_date) + pd.Timedelta('13:30:00')
    train_ts = price_ts[:split_time]
    test_ts = price_ts[split_time:]
    # create strategy
    strat = CorpusStrategy(train_ts, n)
    strat.get_vectorized_signal(test_ts)
    # accept signal and produce position
    port = EqualWeightPort()
    port.get_position(strat.signals.iloc[(n-1):])
    # backtest
    bt = Backtest(test_ts[port.positions.index], port.positions)
    stats = bt.output_summary_stats(periods=1)
    bt.equity_curve.plot(figsize=(20,15), fontsize=15)
    multi_ec.savefig()
    plt.clf()
    res.append((file_date, stats))
multi_price.close()
multi_ec.close()
res = list(zip(*res))
res_df = pd.DataFrame(list(res[1]), res[0])
res_df.to_csv(os.path.join(output_dir, 'corpus_stats_summary.csv'))
###############################################################################


###############################################################################
#******************************** buy-and-hold ********************************
res = []
for file in files:
    file_date = file.split('_')[1]
    concat_data_dir = os.path.join(concat_dir, file)
    cf.read(concat_data_dir)
    price_ts = cf.get_clean_price_ts('MidPrice', trading_time_slots[cft], 
                                     outliers)
    split_time = pd.Timestamp(file_date) + pd.Timedelta('13:30:00')
    train_ts = price_ts[:split_time]
    test_ts = price_ts[split_time:]
    # create strategy
    strat = BuyHoldStrategy()
    strat.get_vectorized_signal(test_ts)
    # accept signal and produce position
    port = EqualWeightPort()
    port.get_position(strat.signals.iloc[(n-1):])
    # backtest
    bt = Backtest(test_ts[port.positions.index], port.positions)
    stats = bt.output_summary_stats(periods=1)
    res.append((file_date, stats))
res = list(zip(*res))
res_df = pd.DataFrame(list(res[1]), res[0])
res_df.to_csv(os.path.join(output_dir, 'bnh_stats_summary.csv'))
###############################################################################


###############################################################################
#******************************** random guess ********************************
res = []
for file in files:
    file_date = file.split('_')[1]
    concat_data_dir = os.path.join(concat_dir, file)
    cf.read(concat_data_dir)
    price_ts = cf.get_clean_price_ts('MidPrice', trading_time_slots[cft],
                                     outliers)
    split_time = pd.Timestamp(file_date) + pd.Timedelta('13:30:00')
    train_ts = price_ts[:split_time]
    test_ts = price_ts[split_time:]
    # create strategy
    strat = RandomGuessStrategy()
    strat.get_vectorized_signal(test_ts)
    # accept signal and produce position
    port = EqualWeightPort()
    port.get_position(strat.signals.iloc[(n-1):])
    # backtest
    bt = Backtest(test_ts[port.positions.index], port.positions)
    stats = bt.output_summary_stats(periods=1)
    res.append((file_date, stats))
res = list(zip(*res))
res_df = pd.DataFrame(list(res[1]), res[0])
res_df.to_csv(os.path.join(output_dir, 'rg_stats_summary.csv'))
###############################################################################
