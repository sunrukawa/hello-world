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
    
    Subclasses need to override the 'get_position' method.
    
    Instance attributes:
        - positions: Positions of a portfolio.
    """
    
    def __init__(self):
        """Initialization method.
        """
        self.positions = None
    
    @abstractmethod
    def get_position(self, signals):
        """Get the position of a portfolio.
        Args:
            signals: Trading signals.
        """
        raise NotImplementedError("Should implement get_position method!")



class EqualWeightPort(Portfolio):
    """Represents an equal weight portfolio.
    
    Subclass of 'Portfolio', which inherits its attributes and overrides 
    'get_position' method.
    """
       
    def get_position(self, signals):
        """Get the position of a portfolio.
        Args:
            signals: Trading signals (pd.Series).
        """
        signals = signals.dropna()
        self.positions = pd.Series(np.where(signals=='2', 1, -1), 
                                   index=signals.index, name='position')
