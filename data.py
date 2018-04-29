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
    
    Subclasses need to override the 'read' and 'get_clean_price_ts' methods 
    to customize behaviors.
    
    Instance attributes:
        - data: A dataset.
    """
    
    def __init__(self):
        """Initialization method.
        """
        self.data = None
    
    @abstractmethod
    def read(self, data_dir): 
        """Read data.
        """
        raise NotImplementedError("Should implement read method first!")

    @abstractmethod
    def get_clean_price_ts(self, price_col): 
        """Clean data and return a price time series.
        """
        raise NotImplementedError("Should implement clean method first!")
        
        

class OHLCData(Data):
    """Represents a dataset formatted with 'open', 'high', 'low', 'close', 
    'adj close', 'volume'.
    
    Subclass of 'Data', which inherits its attributes and overrides 'read' and 
    'get_clean_price_ts' methods.
    """
    
    def read(self, data_dir):
        """Read data.    
        Args:
            data_dir: Directory of OHLC data.
        """
        self.data = pd.read_csv(data_dir, index_col=[0], parse_dates=[0], 
                                na_values=['null'])
    
    def get_clean_price_ts(self, price_col):
        """Clean data and return a price time series.        
        Args:
            price_col: The column is used to clean the dataset and to be 
                returned.       
        Returns:
            A pd.Series represent clean 'Adj Close' price ts.
        """
        df = self.data.copy()
        df = df.sort_index().dropna(subset=[price_col])
        df = df[df[price_col].shift()!=df[price_col]]
        return df[price_col]
    


class CommodityFutureData(Data):
    """Represents a commodity future dataset.
    
    Subclass of 'Data', which inherits its attributes and overrides 'read' and 
    'get_clean_price_ts' methods.
    
    Instance attributes:
        - trading_time_slots: The trading time slots.
        - date: The date of the concatenated dataset.
    """
    
    @classmethod
    def get_acc_volume(cls, file):
        """Read the last line of a specific file and get its accumulative 
        volume.
        Args:
            file: A commodity future data file.
        Returns:
            A float number represents AccVolume in the last line of data file.
        """  
        with open(file, 'rb') as f:
            f.seek(-2, os.SEEK_END)     
            while f.read(1) != b'\n':   
                if f.seek(-2, os.SEEK_CUR) == 0:
                    return None
            return float(f.readline().decode().split(',')[8])
    
    @classmethod
    def get_main_contracts(cls, day_data_dir):
        """Get the main contract for each date.
        Args:
            data_dir: The directory of day commodity future data.
        Returns:
            List of lists, each sublist is in the format of 
                [date, main contract].
        """
        vol_all = defaultdict(list)
        # for each date, collect all the contracts and their AccVolume
        # note: 1) empty file; 2) one-line file
        for filename in os.listdir(day_data_dir):
            file = os.path.join(day_data_dir, filename)
            if not filename.startswith('.') and os.stat(file).st_size != 0:
                accum_vol = cls.get_acc_volume(file)
                if accum_vol != None:
                    contract, date = filename.split('.')[0].split('_')
                    vol_all[date].append((contract, accum_vol))
        # get the main contract for each date
        mc = [[d, max(vols, key=itemgetter(1))[0]] for d, vols 
               in vol_all.items()]            
        return sorted(mc, key=itemgetter(0))
    
    @classmethod
    def concat_data(cls, main_contracts, output_dir, data_dir, day_dir='day', 
                   night_dir='night'):
        """Concatenate day and night future data.
        Args:
            main_contracts: List of lists or ndarray, with dim(-1, 2).
            output_dir: The output directory of concatenated commodity future 
                dataset.
            data_dir: The directory of commodity future data.
            day_dir: The relative path of day commodity future data.
            night_dir: The relative path of night commodity future data.
        """
        for date, mc in main_contracts:
            df_day = pd.read_csv(os.path.join(data_dir, day_dir, 
                                              '{}_{}.csv'.format(mc, date)))
            df_night = pd.read_csv(os.path.join(data_dir, night_dir, 
                                                '{}_{}.csv'.format(mc, date)))
            df = pd.concat([df_day, df_night], ignore_index=True)
            df.to_csv(os.path.join(output_dir, 
                                   '{}_{}_concat.csv'.format(mc, date)))
    
    def __init__(self, trading_time_slots, outliers):
        """Initialization method.
        """
        super().__init__()
        self.trading_time_slots = trading_time_slots
        self.outliers = outliers
        self.date = None
    
    def read(self, concat_data_dir):
        """Read the data from an already concatenated file.
        Args:
            concat_data_dir: The directory of a concatenated commodity future 
                data.
        """
        self.data = pd.read_csv(concat_data_dir)
        self.date = concat_data_dir.split('_')[1]
        
    def get_clean_price_ts(self, price_col):
        """Clean data and return a price time series.        
        Args:
            price_col: The column is used to clean the dataset and to be 
                returned.       
        Returns:
            A pd.Series represent clean 'MidPrice' price ts.
        """
        df = self.data.copy()
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        df['TS'] = df['Date'] + pd.to_timedelta(df['UpdateTime'])
        df = df.set_index('TS')
        timestamps = self._get_timestamps()
        df = df[(df.index>=timestamps[0])&(df.index<=timestamps[-1])]
        df = df[~((df.index>timestamps[1])&(df.index<timestamps[2]))]
        df = df[~((df.index>timestamps[3])&(df.index<timestamps[4]))]
        df = df[~((df.index>timestamps[5])&(df.index<timestamps[6]))]  
        df = df[~df['BidPrice1'].isin(self.outliers)]
        df = df[~df['AskPrice1'].isin(self.outliers)]
        df[price_col] = (df['BidPrice1'] + df['AskPrice1']) / 2
        df = df[df[price_col].shift()!=df[price_col]]
        return df[price_col]
            
    def _get_timestamps(self):
        """Convert 'date' and 'trading_time_slots' to corresponding Timestamps.
        Returns:
            A list of Timestamps.
        """
        timestamps = []
        for time in self.trading_time_slots:
            timestamps.append(pd.Timestamp(self.date)+pd.Timedelta(time))
        timestamps[-1] = timestamps[-1] + pd.Timedelta(1, unit='D')
        return timestamps
    