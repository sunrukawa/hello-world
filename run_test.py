# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:46:25 2018

@author: rl17174
"""

import os
os.chdir(r'C:\Users\bs40027\Desktop\test_ran_revised')

from strategy import CorpusStrategy
from data  import OHLCData
from portfolio import EqualWeightPort
from backtest import Backtest


data_dir = r'C:\Users\bs40027\Desktop\test_ran_revised\000001.SS.csv'
# Process data
SHcomp = OHLCData(data_dir)
SHcomp.read()
SHcomp.clean('Adj Close')
SHcomp.split((('1995', '2004'), ('2005', '2013')))

# Create strategy
strat = CorpusStrategy(SHcomp.train['Adj Close'], 6)
strat.get_vectorized_signal(SHcomp.test['Adj Close'])

# Portfolio accepts signal and produce position
port = EqualWeightPort()
port.get_position(strat.signals)

# Backtest
my_bt = Backtest(SHcomp.test['Adj Close'][port.positions.index], 
                 port.positions)
stats = my_bt.output_summary_stats(rf=0.02)

print(stats)
