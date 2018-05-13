# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:46:25 2018

@author: rl17174
"""

import os
os.chdir(r'C:\Users\bs40027\Desktop\qsproject\bin')
import pandas as pd

from strategy import CorpusStrategy, BuyHoldStrategy, RandomGuessStrategy
from strategy import SimpleMeanRevertingStrategy
from strategy import SimpleMomentumStrategy
from data import CommodityFutureData
from portfolio import EqualWeightPort
from backtest import Backtest
from constants import trading_time_slots, commission_fees

def preprocess_future_data(future_product, future_data_dir, concat_dir):
    """Preprocess the future data, through finding main contracts, getting both
    avaialble day and night data, and concatenating them.
    Args:
        future_product: Future product type, such as 'ag'.
        future_data_dir: Raw future data directory.
        concat_dir: The directory of where to store concatenated data.
    """
    data_dir = os.path.join(future_data_dir, future_product)
    day_dir = os.path.join(data_dir, 'day')
    night_dir = os.path.join(data_dir, 'night')
    concat_dir = os.path.join(data_dir, concat_dir)
    if not os.path.exists(concat_dir):
        os.makedirs(concat_dir)
    mcs = CommodityFutureData.get_main_contracts(day_dir)
    mc_files = CommodityFutureData.get_avail_main_contracts_files(mcs, 
                                                                  night_dir)
    CommodityFutureData.concat_data(mc_files, concat_dir, day_dir, night_dir)
 
def get_input_files(input_dir):
    """Get all input files from a given directory.
    Args:
        input_dir: The directory of input files.
    """
    files = os.listdir(input_dir)
    files = sorted(files)
    if files[0].startswith('.'):
        files = files[1:]
    return files

def get_corpus_strategy(train_ts, n):
    """Get a corpus strategy object.
    Args:
        train_ts: Train data time series.
        n: A hyper parameter, representing number of order.
    Returns:
        A corpus strategy object.
    """
    return CorpusStrategy(train_ts, n)

def get_buy_n_hold_strategy():
    """Get a buy-and-hold strategy object.
    Returns:
        A buy-and-hold strategy object.
    """
    return BuyHoldStrategy()

def get_random_guess_strategy():
    """Get a random guess strategy object.
    Returns:
        A random guess strategy object.
    """
    return RandomGuessStrategy()

def get_simple_mean_reverting_strategy():
    """Get a simple mean reverting strategy object.
    Returns:
        A simple mean reverting strategy object.
    """
    return SimpleMeanRevertingStrategy()

def get_simple_momentum_strategy():
    """Get a simple momentum strategy object.
    Returns:
        A simple momentum strategy object.
    """
    return SimpleMomentumStrategy()

def get_train_test_ts(input_dir, file, outliers):
    """Clean concatenated data and generate train and test price time series.
    Args:
        input_dir: The directory of input file.
        file: File name.
        outliers: Values which can be considered as outliers.
    Returns:
        Train and test price time series.
    """
    concat_data_dir = os.path.join(input_dir, file)
    cf.read(concat_data_dir)
    price_ts = cf.get_clean_price_ts('MidPrice', 
                                     trading_time_slots[cf.future_product], 
                                     outliers)
    split_time = pd.Timestamp(file.split('_')[1]) + pd.Timedelta('13:30:00')
    train_ts = price_ts[:split_time]
    test_ts = price_ts[split_time:]
    return train_ts, test_ts

def get_stats_from_strategy(strategy_type, train_ts, test_ts, commission_fee, 
                            compound, rf, periods):
    """Calculate backtesting statistics for a specific strategy.
    Args:
        strategy_type: A specific strategy.
        train_ts: Train data time series.
        test_ts: Test data time series.
        commission_fee: A number represents the commission fee.
        compound: A bool variable indicating whehter calculates compound 
            return or not.
        rf: Risk free rate.
        periods: Scaling parameter to calculate annualized sharpe ratio.
    Returns:
        A summary statistics.
    """
    if strategy_type == 'corpus':
        strat = strats[strategy_type](train_ts, n)
    else:
        strat = strats[strategy_type]()
    strat.get_vectorized_signals(test_ts)
    # accept signal and produce position
    port = EqualWeightPort()
    port.get_position(strat.signals.iloc[(n-1):])
    # backtest
    bt = Backtest(test_ts[port.positions.index], port.positions)
    stats = bt.output_summary_stats(rf=rf, periods=periods, compound=compound, 
                                    commission_fee=commission_fee)
    return stats

def get_result(cf, future_data_dir, concat_dir, output_dir, strategy_types, n, 
               commission_fee=0, compound=False, rf=0, periods=1):
    """Calculate the backtesting results.
    Args:
        cf: A commodity future object.
        future_product: Future product type, such as 'ag'.
        concat_dir: The directory of where to store concatenated data.
        output_dir: The output directory.
        strategy_types: A list specific strategies.
        n: A hyper parameter, representing number of order.
        commission_fee: A number represents the commission fee.
        compound: A bool variable indicating whehter calculates compound 
            return or not.
        rf: Risk free rate.
        periods: Scaling parameter to calculate annualized sharpe ratio.
    """
    input_dir = os.path.join(future_data_dir, cf.future_product, concat_dir)
    files = get_input_files(input_dir)
    res_all = {strategy_type:[] for strategy_type in strategy_types}
    outliers = (-1, 0)
    for file in files:
        train_ts, test_ts = get_train_test_ts(input_dir, file, outliers)
        # create strategy
        for strategy_type in strategy_types:
            stats = get_stats_from_strategy(strategy_type, train_ts, test_ts, 
                                         commission_fee, compound, rf, periods)
            res_all[strategy_type].append((file.split('_')[1], stats))
    for strategy_type, res in res_all.items():
        res = list(zip(*res))
        res_df = pd.DataFrame(list(res[1]), res[0])
        res_df.to_csv(os.path.join(output_dir, cf.future_product, 
                                   '{}_summary.csv'.format(strategy_type)))

strats = {'corpus': get_corpus_strategy,
          'bnh': get_buy_n_hold_strategy,
          'rg': get_random_guess_strategy,
          'smr': get_simple_mean_reverting_strategy,
          'sm': get_simple_momentum_strategy}

if __name__ == '__main__':
#    future_products = ['ag', 'bu', 'rb', 'ru', 'zn']
    future_products = ['bu', 'rb']
    future_data_dir = '../data/future data'
    concat_dir = 'subconcat'
    output_dir = '../output'
    strategy_types = ['corpus']
    n = 3
    for future_product in future_products:
#        preprocess_future_data(future_product, future_data_dir, concat_dir)
        cf = CommodityFutureData(future_product)
        commission_fee = commission_fees[cf.future_product]  
        get_result(cf, future_data_dir, concat_dir, output_dir, strategy_types, 
                   n, commission_fee=commission_fee)
