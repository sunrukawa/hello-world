# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:39:33 2018

@author: rl17174
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from collections import Counter
from operator import itemgetter

class Strategy(ABC):
    """Represents a strategy.
    """
    
    @abstractmethod
    def get_signal(self): 
        """
        Provides the mechanisms to calculate the list of signals.
        """
        raise NotImplementedError("Should implement get_signal method first!")
        
        
        
class CorpusStrategy(Strategy):
    """Represents NLP strategy.
    """
    
    def __init__(self, init_train_price_ts, n):
        """Initialize strategy instance.
        """
        self.n = n
        # This corpus may be updated later as data flow in
        self.corpus_dict = self._get_corpus(self._get_symbol(
                                                          init_train_price_ts))
        self.signal = None
        self.signals = None
        
    def _get_corpus(self, ts_sym):
        """Establish the corpus dictionary.
        Args:
            ts_sym: A string represent symbol and ndarray of symbols
        Returns:
            Dictionary, contains {symbol: counts}.
        """
        corpus_dict = Counter(ts_sym[i:i+self.n] for i in \
                              range(len(ts_sym)-self.n+1))
        return corpus_dict
        
    def _get_symbol(self, price_ts):
        """Encode information such as up and down into symbols.
        Args:
            ts: pd series, represents mid or last price time series.
        Returns:
            A string represent symbol and ndarray of symbols (optional).
        """
        ret = pd.Series(price_ts).pct_change().dropna()
        # note: tuning parameter 'm' is hard-coded as '0', '1', '2'
        symbol_arr = np.where(ret==0, '0', np.where(ret<0, '1', '2'))
        ts_sym = ''.join(symbol_arr.tolist())
        return ts_sym     

    def update_corpus(self, train_price_ts):
        self.corpus_dict = self._get_corpus(self._get_symbol(train_price_ts))
        
    def _retrieve_from_corpus(self, symbol_str):
        symbol_freq = {symbol:self.corpus_dict.get(symbol_str+symbol, 0)
                           for symbol in '012'}
        signal, freq = max(symbol_freq.items(), key=itemgetter(1))
        #???? need to consider freq equal
        return signal
        
    def get_vectorized_signal(self, test_price_ts):
        """vectorized calculation for signal"""
        signals = pd.Series(index=test_price_ts.index, name='signal')
        symbol_str = self._get_symbol(test_price_ts)
        for i in range(len(symbol_str)-self.n+2):
            signals.iloc[i+self.n-1] = self._retrieve_from_corpus(
                                                      symbol_str[i:i+self.n-1])
        self.signals = signals
    
    def get_signal(self, test_price_ts):
        symbol_str = self._get_symbol(test_price_ts)
        self.signal = self._retrieve_from_corpus(symbol_str)
