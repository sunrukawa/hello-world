# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 11:31:44 2018

@author: bs40027
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir(r'C:\Users\bs40027\Desktop\qsproject')

trading_time_slots = {'ag':('09:00:00', '10:15:00', '10:30:00', '11:30:00', 
                            '13:30:00', '15:00:00', '21:00:00', '02:30:00'),
                      'bu':('09:00:00', '10:15:00', '10:30:00', '11:30:00', 
                            '13:30:00', '15:00:00', '21:00:00', '23:00:00'),
                      'rb':('09:00:00', '10:15:00', '10:30:00', '11:30:00', 
                            '13:30:00', '15:00:00', '21:00:00', '23:00:00'),
                      'ru':('09:00:00', '10:15:00', '10:30:00', '11:30:00', 
                            '13:30:00', '15:00:00', '21:00:00', '23:00:00'),
                      'zn':('09:00:00', '10:15:00', '10:30:00', '11:30:00', 
                            '13:30:00', '15:00:00', '21:00:00', '01:00:00')}
            
def get_timestamps(trading_time_slots, date):
        """Convert 'date' and 'trading_time_slots' to corresponding Timestamps.
        Returns:
            A list of Timestamps.
        """
        timestamps = []
        for time in trading_time_slots:
            timestamps.append(pd.Timestamp(date)+pd.Timedelta(time))
        if pd.Timedelta(trading_time_slots[-1]) < pd.Timedelta('12:00:00'):
            timestamps[-1] = timestamps[-1] + pd.Timedelta(1, unit='D')
        return timestamps


outliers = (-1, 0)
df = pd.read_csv('./data/future data/ag/concat/ag1606_20160104_concat.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
df['TS'] = df['Date'] + pd.to_timedelta(df['UpdateTime'])
df = df.set_index('TS')
timestamps = get_timestamps(trading_time_slots['ag'], '20160104')
start, end = timestamps[2], timestamps[2]+pd.Timedelta(0.25,unit='h')


fig, axes = plt.subplots(2, 4, figsize=(32, 14))

df['mid_price'] = (df['BidPrice1'] + df['AskPrice1']) / 2
df['mid_price'].plot(ax=axes[0, 0], fontsize=20)
axes[0, 0].set_ylabel('Mid-Price', fontsize=40)
axes[0, 0].margins(y=1)
axes[0, 0].set_xlabel('')

df = df[~df['BidPrice1'].isin(outliers)]
df = df[~df['AskPrice1'].isin(outliers)]
df['mid_price'].plot(ax=axes[0, 1], fontsize=20)
axes[0, 1].set_xlabel('')
#axes[0, 1].set_ylabel('Price', fontsize=20)
axes[0, 1].axvline(x=start, color='red', linestyle='dashed')
axes[0, 1].axvline(x=end, color='red', linestyle='dashed')

df = df[(df.index>=timestamps[0])&(df.index<=timestamps[-1])]
df = df[~((df.index>timestamps[1])&(df.index<timestamps[2]))]
df = df[~((df.index>timestamps[3])&(df.index<timestamps[4]))]
df = df[~((df.index>timestamps[5])&(df.index<timestamps[6]))] 
t = df[start:end]['mid_price'].copy()
t.plot(ax=axes[0, 2], fontsize=20)
axes[0, 2].set_xlabel('')
t = t[t.shift() != t]
t.reset_index().plot(ax=axes[0, 3], legend=False, fontsize=20)
#fig.tight_layout()
#plt.savefig('fig1.png', dpi=300)


outliers = (-1, 0)
df = pd.read_csv('./data/future data/ag/concat/ag1606_20160104_concat.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
df['TS'] = df['Date'] + pd.to_timedelta(df['UpdateTime'])
df = df.set_index('TS')
timestamps = get_timestamps(trading_time_slots['ag'], '20160104')

#fig, axes = plt.subplots(2, 2, figsize=(24, 12))

df['LastPrice'].plot(ax=axes[1, 0], fontsize=20)
axes[1, 0].set_ylabel('Last-Price', fontsize=40)
axes[1, 0].margins(y=1)
axes[1, 0].set_xlabel('')

df = df[~df['LastPrice'].isin(outliers)]
df['LastPrice'].plot(ax=axes[1, 1], fontsize=20)
axes[1, 1].set_xlabel('')
#axes[1, 1].set_ylabel('Price', fontsize=20)
axes[1, 1].axvline(x=start, color='red', linestyle='dashed')
axes[1, 1].axvline(x=end, color='red', linestyle='dashed')

df = df[(df.index>=timestamps[0])&(df.index<=timestamps[-1])]
df = df[~((df.index>timestamps[1])&(df.index<timestamps[2]))]
df = df[~((df.index>timestamps[3])&(df.index<timestamps[4]))]
df = df[~((df.index>timestamps[5])&(df.index<timestamps[6]))] 
t = df[start:end]['LastPrice'].copy()
t.plot(ax=axes[1, 2], fontsize=20)
axes[1, 2].set_xlabel('')
t = t[t.shift() != t]
t.reset_index().plot(ax=axes[1, 3], legend=False, fontsize=20)
fig.tight_layout()
plt.savefig('fig.png', dpi=360)







import numpy as np
def calc_sharpe_ratio(rf, periods, pnl):
    excess_rets = pnl - rf/periods
    return np.sqrt(periods)*np.mean(excess_rets)/np.std(excess_rets)
    
def calc_max_drawdown(equity_curve):
    return 1-min(equity_curve/equity_curve.cummax())
    
def calc_max_drawdown_duration(equity_curve):
    dd = equity_curve-equity_curve.cummax()
    dd.index = range(len(dd))
    idx = np.where(dd==0)[0]
    # might double check
    if len(idx) == 1:
        return -1
    mdd = np.diff(idx).max()
    if mdd == 1:
        return 0
    else:
        return mdd

import matplotlib as mpl
#mpl.rcParams.update({'font.size': 17})
mpl.rcParams.update(mpl.rcParamsDefault)
df = pd.read_csv('./output/bu/mid price/corpus_summary_bu.csv', index_col=0, 
                 parse_dates=[0])
df['ret'] = df['Total Return'].str.rstrip('%').astype('float') / 100
df['ret'] = df['ret'] - 0.0005 * df['No of Transactions']
df['cum_ret'] = df['ret'].cumsum() + 1

sr = calc_sharpe_ratio(0.02, 250, df['ret'])
mdd = calc_max_drawdown(df['cum_ret'])
ddd = calc_max_drawdown_duration(df['cum_ret'])
tr = df['cum_ret'][-1]
df['Winning Rate'] = df['Winning Rate'].str.rstrip('%').astype(float) / 100
df['Max Drawdown'] = df['Max Drawdown'].str.rstrip('%').astype(float) / 100


avg_wr = (df['Winning Rate'] * df['No of Transactions']).sum() / \
         df['No of Transactions'].sum()
avg_nt = df['No of Transactions'].mean()
avg_ret_per_transaction = tr / df['No of Transactions'].sum()

fig, axes = plt.subplots(2, 4, figsize=(32, 14))

df.rename(columns={'cum_ret':'equity curve', 'ret':'daily return',
                   'Max Drawdown':'max drawdown', 
                   'No of Transactions':'no of transactions',
                   'Winning Rate':'winning rate'}, inplace=True)
df['equity curve'].plot(ax=axes[0, 0], legend=True)
(-df['max drawdown']).plot(secondary_y=True, color='tomato', ax=axes[0, 0], legend=True)

df['equity curve'].plot(ax=axes[0, 1], legend=True)
df['no of transactions'].plot(secondary_y=True, kind='area', color='darkgreen', ax=axes[0, 1], legend=True)

df['equity curve'].plot(ax=axes[0, 2], legend=True)
df['winning rate'].plot(secondary_y=True, linestyle='-', marker='.', 
                        color='purple', ax=axes[0, 2], legend=True, ms=8)

df['daily return'].plot(kind='hist', grid=True, bins=30, ax=axes[0, 3], legend=True)
df['daily return'].plot(secondary_y=True, kind='density', grid=True, ax=axes[0, 3])



df = pd.read_csv('./output/bu/corpus_summary_bu.csv', index_col=0, 
                 parse_dates=[0])
df['ret'] = df['Total Return'].str.rstrip('%').astype('float') / 100
df['ret'] = df['ret'] - 0.0009 * df['No of Transactions']
df['cum_ret'] = df['ret'].cumsum() + 1

sr = calc_sharpe_ratio(0.02, 250, df['ret'])
mdd = calc_max_drawdown(df['cum_ret'])
ddd = calc_max_drawdown_duration(df['cum_ret'])
tr = df['cum_ret'][-1]
df['Winning Rate'] = df['Winning Rate'].str.rstrip('%').astype(float) / 100
df['Max Drawdown'] = df['Max Drawdown'].str.rstrip('%').astype(float) / 100

avg_wr = (df['Winning Rate'] * df['No of Transactions']).sum() / \
         df['No of Transactions'].sum()
avg_nt = df['No of Transactions'].mean()
avg_ret_per_transaction = tr / df['No of Transactions'].sum()

df.rename(columns={'cum_ret':'equity curve', 'ret':'daily return',
                   'Max Drawdown':'max drawdown', 
                   'No of Transactions':'no of transactions',
                   'Winning Rate':'winning rate'}, inplace=True)
df['equity curve'].plot(ax=axes[1, 0], legend=True)
(-df['max drawdown']).plot(secondary_y=True, color='tomato', ax=axes[1, 0], legend=True)

df['equity curve'].plot(ax=axes[1, 1], legend=True)
df['no of transactions'].plot(secondary_y=True, kind='area', color='darkgreen', ax=axes[1, 1], legend=True)

df['equity curve'].plot(ax=axes[1, 2], legend=True)
df['winning rate'].plot(secondary_y=True, linestyle='-', marker='.', 
                        color='purple', ax=axes[1, 2], legend=True, ms=8)

df['daily return'].plot(kind='hist', grid=True, bins=30, ax=axes[1, 3], legend=True)
df['daily return'].plot(secondary_y=True, kind='density', grid=True, ax=axes[1, 3])


fig.tight_layout()
plt.savefig('fig1.png', dpi=350)



#df2 = pd.read_csv('./output/bu/bu_mp_1d.csv', index_col=['date'], 
#                  parse_dates=['date'])
#df2['equity_curve'] = (df2['pnl'] - 0.0005).cumsum() + 1
#df2['equity_curve'].reset_index().plot(ax=axes[0, 1])
#df2['pnl'].hist(bins=20, ax=axes[1, 1])
#fig.tight_layout()
#plt.savefig('fig2.png')
#
#df2 = pd.read_csv('./output/bu/bu_lp_1d.csv', index_col=['date'], 
#                  parse_dates=['date'])
#df2['equity_curve'] = (df2['pnl'] - 0.001).cumsum() + 1
#df2['equity_curve'].reset_index().plot(ax=axes[0, 1])
#df2['pnl'].hist(bins=20, ax=axes[1, 1])
#fig.tight_layout()
#plt.savefig('fig4.png')
#

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(28, 21))
ax1 = plt.subplot2grid((8, 9), (0, 0), rowspan=4, colspan=3)
ax2 = plt.subplot2grid((8, 9), (4, 0), rowspan=4, colspan=3)
ax3 = plt.subplot2grid((8, 9), (0, 3), rowspan=2, colspan=3)
ax4 = plt.subplot2grid((8, 9), (0, 6), rowspan=2, colspan=3)
ax5 = plt.subplot2grid((8, 9), (2, 3), rowspan=2, colspan=3)
ax6 = plt.subplot2grid((8, 9), (2, 6), rowspan=2, colspan=3)
ax7 = plt.subplot2grid((8, 9), (4, 3), rowspan=2, colspan=3)
ax8 = plt.subplot2grid((8, 9), (4, 6), rowspan=2, colspan=3)
ax9 = plt.subplot2grid((8, 9), (6, 3), rowspan=2, colspan=3)
ax10 = plt.subplot2grid((8, 9), (6, 6), rowspan=2, colspan=3)

outliers = (-1, 0)
df = pd.read_csv('./data/future data/ag/concat/ag1606_20160104_concat.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
df['TS'] = df['Date'] + pd.to_timedelta(df['UpdateTime'])
df = df.set_index('TS')
timestamps = get_timestamps(trading_time_slots['ag'], '20160104')
df = df[['LastPrice', 'BidPrice1', 'AskPrice1']]
df = df[~df['LastPrice'].isin(outliers)]
df = df[~df['BidPrice1'].isin(outliers)]
df = df[~df['AskPrice1'].isin(outliers)]
df.plot(ax=ax1, fontsize=18)
ax1.legend(fontsize=18)
df.iloc[10000:11000, :].plot(ax=ax2, fontsize=18)
ax2.legend(fontsize=18)


df = pd.read_csv('./output/bu/mid price/corpus_summary_bu.csv', index_col=0, 
                 parse_dates=[0])
df['ret'] = df['Total Return'].str.rstrip('%').astype('float') / 100
df['ret1'] = df['ret'] - 0.0001 * df['No of Transactions']
df['cum_ret1'] = df['ret1'].cumsum() + 1
df['cum_ret1'].plot(ax=ax3, fontsize=17)
ax3.legend(['equity curve (slippage_cost=1bps)'], fontsize=17)
df['ret2'] = df['ret'] - 0.0003 * df['No of Transactions']
df['cum_ret2'] = df['ret2'].cumsum() + 1
df['cum_ret2'].plot(ax=ax5, fontsize=17)
ax5.legend(['equity curve (slippage_cost=3bps)'], fontsize=17)
df['ret3'] = df['ret'] - 0.0005 * df['No of Transactions']
df['cum_ret3'] = df['ret3'].cumsum() + 1
df['cum_ret3'].plot(ax=ax7, fontsize=17)
ax7.legend(['equity curve (slippage_cost=5bps)'], fontsize=17)
df['ret4'] = df['ret'] - 0.0006 * df['No of Transactions']
df['cum_ret4'] = df['ret4'].cumsum() + 1
df['cum_ret4'].plot(ax=ax9, fontsize=17)
ax9.legend(['equity curve (slippage_cost=6bps)'], fontsize=17)

df = pd.read_csv('./output/bu/corpus_summary_bu.csv', index_col=0, 
                 parse_dates=[0])
df['ret'] = df['Total Return'].str.rstrip('%').astype('float') / 100
df['ret1'] = df['ret'] - 0.0005 * df['No of Transactions']
df['cum_ret1'] = df['ret1'].cumsum() + 1
df['cum_ret1'].plot(ax=ax4, fontsize=17)
ax4.legend(['equity curve (slippage_cost=5bps)'], fontsize=17)
df['ret2'] = df['ret'] - 0.0007 * df['No of Transactions']
df['cum_ret2'] = df['ret2'].cumsum() + 1
df['cum_ret2'].plot(ax=ax6, fontsize=17)
ax6.legend(['equity curve (slippage_cost=7bps)'], fontsize=17)
df['ret3'] = df['ret'] - 0.0009 * df['No of Transactions']
df['cum_ret3'] = df['ret3'].cumsum() + 1
df['cum_ret3'].plot(ax=ax8, fontsize=17)
ax8.legend(['equity curve (slippage_cost=9bps)'], fontsize=17)
df['ret4'] = df['ret'] - 0.001 * df['No of Transactions']
df['cum_ret4'] = df['ret4'].cumsum() + 1
df['cum_ret4'].plot(ax=ax10, fontsize=17)
ax10.legend(['equity curve (slippage_cost=10bps)'], fontsize=17)

fig.tight_layout()
plt.savefig('fig2.png', dpi=300)




fig, axes = plt.subplots(2, 2, figsize=(28, 20))
mid = os.path.join(os.getcwd(), './output/mid_price_results')
files = os.listdir(mid)
df = pd.read_csv(os.path.join(mid, files[0]), parse_dates=[0], index_col=0)
df['Winning Rate'] = df['Winning Rate'].str.rstrip('%').astype(float) / 100
df['Total Return'] = df['Total Return'].str.rstrip('%').astype(float) / 100
df['Max Drawdown'] = df['Max Drawdown'].str.rstrip('%').astype(float) / 100
df.columns = [col+'_{}'.format(files[0].split('_')[-1].split('.')[0]) for col in df.columns]

for file in files[1:]:
    df_temp = pd.read_csv(os.path.join(mid, file), parse_dates=[0], index_col=0)
    df_temp['Winning Rate'] = df_temp['Winning Rate'].str.rstrip('%').astype(float) / 100
    df_temp['Total Return'] = df_temp['Total Return'].str.rstrip('%').astype(float) / 100
    df_temp['Max Drawdown'] = df_temp['Max Drawdown'].str.rstrip('%').astype(float) / 100

    df_temp.columns = [col+'_{}'.format(file.split('_')[-1].split('.')[0]) for col in df_temp.columns]
    df = df.join(df_temp, how='outer')
prods = ['ag', 'bu', 'rb', 'ru', 'zn']
for prod in prods:
    df['equity curve'+' '+prod] = df['Total Return'+'_'+prod].cumsum() + 1
df[['equity curve'+' '+prod for prod in prods]].plot(ax=axes[0, 0], fontsize=20)
axes[0,0].legend(fontsize=20) 
df['month'] = df.index.month
df_gb = df.groupby(['month'])[['Total Return'+'_'+prod for prod in prods]].sum()
import calendar
months = [calendar.month_abbr[num] for num in df_gb.index]
df_gb.index = months
df_gb.columns = ['monthly return '+x.split('_')[-1] for x in df_gb.columns]
df_gb.plot(kind='bar', ax=axes[1, 0], fontsize=20)
axes[1,0].legend(fontsize=20) 
axes[1, 0].set_xlabel('slippage cost = 0', fontsize=23)

df['tr_ag_1'] = df['Total Return_ag'] - 0.000075 * df['No of Transactions_ag']
df['tr_bu_1'] = df['Total Return_bu'] - 0.000550 * df['No of Transactions_bu']
df['tr_rb_1'] = df['Total Return_rb'] - 0.000110 * df['No of Transactions_rb']
df['tr_ru_1'] = df['Total Return_ru'] - 0.000050 * df['No of Transactions_ru']
df['tr_zn_1'] = df['Total Return_zn'] - 0.000075 * df['No of Transactions_zn']

df['adj equity curve ag'] = df['tr_ag_1'].cumsum() + 1
df['adj equity curve bu'] = df['tr_bu_1'].cumsum() + 1
df['adj equity curve rb'] = df['tr_rb_1'].cumsum() + 1
df['adj equity curve ru'] = df['tr_ru_1'].cumsum() + 1
df['adj equity curve zn'] = df['tr_zn_1'].cumsum() + 1

df[['adj equity curve '+prod for prod in prods]].plot(ax=axes[0, 1], fontsize=20)
axes[0,1].legend(fontsize=20)   
df_gb_1 = df.groupby(['month'])[['tr'+'_'+prod+'_1' for prod in prods]].sum()
df_gb_1.index = months
df_gb_1.columns = ['monthly return '+x.split('_')[1] for x in df_gb_1.columns]
df_gb_1.plot(kind='bar', ax=axes[1, 1], fontsize=20)
axes[1,1].legend(fontsize=20) 
axes[1, 1].set_xlabel('sc_ag=0.75bps, sc_bu=5.5bps, sc_rb=1.1bps, sc_ru=0.5bps, sc_zn=0.75bps', fontsize=23)

fig.tight_layout()
plt.savefig('fig3.png', dpi=300)
