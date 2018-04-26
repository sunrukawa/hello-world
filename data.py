# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:01:45 2018

@author: rl17174
"""

from abc import ABC, abstractmethod
import os
import pandas as pd
from collections import defaultdict
from operator import itemgetter

class Data(ABC):
    """Represents a dataset.
    """
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = None
        self.train = None
        self.test = None
    
    @abstractmethod
    def read(self): 
        """
        Read data.
        """
        raise NotImplementedError("Should implement read method first!")

    @abstractmethod
    def clean(self, price_col): 
        """
        Clean data.
        """
        raise NotImplementedError("Should implement clean method first!")
        
    @abstractmethod
    def split(self, cutoff_date): 
        """
        Split data to training & testing.
        """
        raise NotImplementedError("Should implement split method first!")
        
    

class OHLCData(Data):
    """Represents a dataset formatted with 'open', 'high', 'low', 'close', 
    'adj close', 'volume'.
    """
    
    def read(self):
        self.data = pd.read_csv(self.data_dir, index_col = [0], 
                                parse_dates=[0], na_values=['null'])
    
    def clean(self, price_col):
        self.data = self.data.sort_index().dropna(subset=[price_col])
        self.data = self.data[self.data[price_col].shift()
                              !=self.data[price_col]]
    
    def split(self, cutoff_date): 
        self.train = self.data[cutoff_date[0][0]:cutoff_date[0][1]].copy()
        self.test  = self.data[cutoff_date[1][0]:cutoff_date[1][1]].copy()
    
    

class FutureData(Data):
    """Represents a dataset from the future data.
    """
    
    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.chain_data = None
    
    def read(self):
        self.data = pd.read_csv(self.data_dir)
        
    def clean(self, price_col):
        pass
    
    def split(self, cutoff_date): 
        pass
    
    def chain_data(self, chain_data_dir):
        """Put data together.
        Args:
            main_contracts: list of lists or ndarray, with dim(-1, 2).
            data_dir: data directory.
        Returns:
            DataFrame: chained data.
        """
        main_contracts = self._get_main_contracts()
        df_all = (pd.read_csv(os.path.join(chain_data_dir, 
                 '{}_{}.csv'.format(mc, date))) for date, mc in main_contracts)
        self.chain_data = pd.concat(df_all, ignore_index=True)
    
    def _get_acc_volume(self, file):
        """Read the last line of a specific file and get its accumulative 
        volume.
        Args:
            file: commodity future data file.
        Returns:
            A float number represents AccVolume in the last line of data file.
        """  
        with open(file, 'rb') as f:
            f.seek(-2, os.SEEK_END)     
            while f.read(1) != b'\n':   
                if f.seek(-2, os.SEEK_CUR) == 0:
                    return None
            return float(f.readline().decode().split(',')[8])
        
    def _get_main_contracts(self):
        """Get the main contract for each date.
        Args:
            data_dir: data directory.
        Returns:
            List of lists, each sublist is in the format of 
            [date, main contract].
        """
        vol_all = defaultdict(list)
        # for each date, collect all the contracts and their AccVolume
        # note: 1) empty file; 2) one-line file
        for filename in os.listdir(self.data_dir):
            file = os.path.join(self.data_dir, filename)
            if not filename.startswith('.') and os.stat(file).st_size != 0:
                accum_vol = self._get_acc_volume(file)
                if accum_vol != None:
                    contract, date = filename.split('.')[0].split('_')
                    vol_all[date].append((contract, accum_vol))
        # get the main contract for each date
        mc = [[d, max(vols, key=itemgetter(1))[0]] for d, vols 
               in vol_all.items()]            
        main_contracts = sorted(mc, key=itemgetter(0))
        return main_contracts
    