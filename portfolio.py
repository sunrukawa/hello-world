# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 19:11:30 2018

@author: rl17174
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class Portfolio(ABC):
    """Represents a portfolio.
    """
    
    def __init__(self):
        self.positions = None
    
    @abstractmethod
    def get_position(self, signals):
        raise NotImplementedError("Should implement get_position method!")



class EqualWeightPort(Portfolio):
    """Represents an equal weight portfolio.
    """
       
    def get_position(self, signals):
        signals = signals.dropna()
        self.positions = pd.Series(np.where(signals=='2', 1, -1), 
                                   index=signals.index, name='position')
