# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 10:20:12 2018

@author: bs40027
"""

import pandas as pd
import numpy as np
import os
from operator import itemgetter
import matplotlib.pyplot as plt


data_dir = r'C:\Users\bs40027\Desktop\future data'
day_dir = 'day'
night_dir = 'night'

day_files = os.listdir(os.path.join(data_dir, day_dir))
night_files = os.listdir(os.path.join(data_dir, day_dir))

aug_day_files = [os.path.join(day_dir, file) for file in day_files]
day_dates = [int(file.split('_')[1].split('.')[0]+'0') for file in day_files]
day_files_zip = list(zip(aug_day_files, day_dates))

aug_night_files = [os.path.join(night_dir, file) for file in night_files]
night_dates = [int(file.split('_')[1].split('.')[0]+'1') 
               for file in night_files]
night_files_zip = list(zip(aug_night_files, night_dates))

files = day_files_zip + night_files_zip
files = sorted(files, key=itemgetter(1))

df_all = (pd.read_csv(os.path.join(data_dir, file)) 
          for file, _ in files)
df_all = pd.concat(df_all, ignore_index=True)
df_all['Date'] = pd.to_datetime(df_all['Date'], format='%Y%m%d')

date_1 = '2016-11-15'
date_2 = '2016-11-16'
m_s1 = pd.Timestamp('') 
m_s1 = pd.Timestamp(date_1 + '')


outliers = (-1, 0)
df = df_all[df_all['Date']=='2016-11-15'].copy()
df['TimeStamp'] = df['TimeStamp'].map(lambda x: '{:.0f}'.format(x))
df['TS'] = pd.to_datetime(df['TimeStamp'], unit='ms')
df['TS'] = df['TS'].dt.tz_localize('utc').dt.tz_convert('Asia/Shanghai')
df = df.set_index('TS')
df = df[~df['BidPrice1'].isin(outliers)]
df = df[~df['AskPrice1'].isin(outliers)]

df['MidPrice'] = (df['BidPrice1'] + df['AskPrice1']) / 2

df['MidPrice'].plot(figsize=(30, 20), xticks=df.index[::1000])

x = df['TimeStamp'].astype(float)
y = x-x.shift(1)
np.where(y>10000)
plt.savefig('test.pdf', dpi=1000)

