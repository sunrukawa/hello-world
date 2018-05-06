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
    
    Subclasses need to override the 'get_signal' method.
    """
    
    @abstractmethod
    def get_signal(self, price_ts): 
        """Get the signal from the price ts.
        Args:
            price_ts: A price time series.
        """
        raise NotImplementedError("Should implement get_signal method first!")
        
        
        
class CorpusStrategy(Strategy):
    """Represents the NLP strategy.
    
    Subclass of 'Strategy', which overrides 'get_signal' method.
    
    Instance attributes:
        - n: Number of order.
        - corpus_dict: The corpus dictionary.
        - signal: A single signal.
        - signals: A series of signals (pd.Series).
    """
    
    def __init__(self, init_train_price_ts, n):
        """Initialization method.
        Args:
            init_train_price_ts: Initial train data price time series.
            n: Number of order.
        """
        self.n = n
        # the corpus may be updated later as data flow in
        self.corpus_dict = self._get_corpus(self._get_symbol(
                                                          init_train_price_ts))
        self.signal = None
        self.signals = None
        
    def get_signal(self, test_price_ts):
        """Get the signal from the price ts.
        Args:
            test_price_ts: The test price time series. 
        """
        symbol_str = self._get_symbol(test_price_ts)
        self.signal = self._retrieve_from_corpus(symbol_str)
        
    def get_vectorized_signal(self, test_price_ts):
        """A vectorized version of getting signals from the price ts.
        Args:
            test_price_ts: The test price time series.
        """
        signals = pd.Series(index=test_price_ts.index, name='signal')
        symbol_str = self._get_symbol(test_price_ts)
        for i in range(len(symbol_str)-self.n+2):
            signals.iloc[i+self.n-1] = self._retrieve_from_corpus(
                                                      symbol_str[i:i+self.n-1])
        self.signals = signals

    def update_corpus(self, new_train_price_ts):
        """Update the current corpus dictionary.
        Args:
            new_train_price_ts: The new train price time series.
        """
        self.corpus_dict = self._get_corpus(self._get_symbol(
                                                           new_train_price_ts))
    
    def _get_symbol(self, price_ts):
        """Encode information such as up and down into symbols.
        Args:
            price_ts: pd series, represents mid or last price time series.
        Returns:
            A string represents symbols.
        """
        if isinstance(price_ts, pd.Series):
            ret = price_ts.pct_change().dropna()
        else:
            ret = pd.Series(price_ts).pct_change().dropna()
        symbol_arr = np.where(ret<0, '1', '2')
        symbol_str = ''.join(symbol_arr.tolist())
        return symbol_str
    
    def _get_corpus(self, symbol_str):
        """Establish the corpus dictionary.
        Args:
            symbol_str: A string represents symbols.
        Returns:
            Dictionary, contains {symbol: counts}.
        """
        corpus_dict = Counter(symbol_str[i:i+self.n] for i in
                              range(len(symbol_str)-self.n+1))
        return corpus_dict
    
    def _retrieve_from_corpus(self, symbol_str):
        """Retrieve frequency from corpus dictionary and return the signal with
        largest frequency.
        Args:
            symbol_str: A string represents symbols.
        Returns:
            The predicted signal.
        """
        if len(symbol_str) != self.n-1:
            raise ValueError('The length of price time series is not correct!')
        symbol_freq = {symbol:self.corpus_dict.get(symbol_str+symbol, 0)
                       for symbol in '12'}
        signal, freq = max(symbol_freq.items(), key=itemgetter(1))
        #???? need to consider freq equal
        return signal



class BuyHoldStrategy(Strategy):
    """Represents the buy-and-hold strategy.
    
    Subclass of 'Strategy', which overrides 'get_signal' method.
    
    Instance attributes:
        - signals: A series of signals (pd.Series).
    """
    
    def __init__(self):
        """Initialization method.
        """
        self.signals = None
    
    def get_signal(self, test_price_ts):
        """Get the signal from the price ts.
        Args:
            test_price_ts: The test price time series. 
        """
        pass
        
    def get_vectorized_signal(self, test_price_ts):
        """A vectorized version of getting signals from the price ts.
        Args:
            test_price_ts: The test price time series.
        """
        self.signals = pd.Series('2', index=test_price_ts.index, name='signal')
        


class RandomGuessStrategy(Strategy):
    """Represents the random guess strategy.
    
    Subclass of 'Strategy', which overrides 'get_signal' method.
    
    Instance attributes:
        - signals: A series of signals (pd.Series).
    """
    
    def __init__(self):
        """Initialization method.
        """
        self.signals = None
    
    def get_signal(self, test_price_ts):
        """Get the signal from the price ts.
        Args:
            test_price_ts: The test price time series. 
        """
        pass
        
    def get_vectorized_signal(self, test_price_ts):
        """A vectorized version of getting signals from the price ts.
        Args:
            test_price_ts: The test price time series.
        """
        random_signals = np.random.randint(1, 3, len(test_price_ts))
        random_signals = list(map(str, random_signals))
        self.signals = pd.Series(random_signals, index=test_price_ts.index, 
                                 name='signal')



class SimpleMeanRevertingStrategy(Strategy):
    """Represents the random guess strategy.
    
    Subclass of 'Strategy', which overrides 'get_signal' method.
    
    Instance attributes:
        - signals: A series of signals (pd.Series).
    """
    
    def __init__(self):
        """Initialization method.
        """
        self.signals = None
    
    def get_signal(self, test_price_ts):
        """Get the signal from the price ts.
        Args:
            test_price_ts: The test price time series. 
        """
        pass
        
    def get_vectorized_signal(self, test_price_ts):
        """A vectorized version of getting signals from the price ts.
        Args:
            test_price_ts: The test price time series.
        """
        if isinstance(test_price_ts, pd.Series):
            ret = test_price_ts.pct_change().dropna()
        else:
            ret = pd.Series(test_price_ts).pct_change().dropna()
        signals = pd.Series(index=test_price_ts.index, name='signal')
        signals.iloc[1:] = np.where(ret<0, '2', '1')
        self.signals = signals


class SimpleTrendingStrategy(Strategy):
    """Represents the random guess strategy.
    
    Subclass of 'Strategy', which overrides 'get_signal' method.
    
    Instance attributes:
        - signals: A series of signals (pd.Series).
    """
    
    def __init__(self):
        """Initialization method.
        """
        self.signals = None
    
    def get_signal(self, test_price_ts):
        """Get the signal from the price ts.
        Args:
            test_price_ts: The test price time series. 
        """
        pass
        
    def get_vectorized_signal(self, test_price_ts):
        """A vectorized version of getting signals from the price ts.
        Args:
            test_price_ts: The test price time series.
        """
        if isinstance(test_price_ts, pd.Series):
            ret = test_price_ts.pct_change().dropna()
        else:
            ret = pd.Series(test_price_ts).pct_change().dropna()
        self.signals = pd.Series(np.where(ret>0, '2', '1'), index=ret.index,
                                 name='signal')

