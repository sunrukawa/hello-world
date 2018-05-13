# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:31:45 2018

@author: rl17174
"""

import numpy as np
from collections import OrderedDict

class Backtest:
    """Represents a backtest.
    
    Instance attributes:
        - prices: Price time series.
        - positions: Portfolio position series.
        - pnl: P&L values (pd.Series).
        - equity_curve: Cumulative returns (pd.Series). 
    """
    
    def __init__(self, prices, positions):
        """Initialization method.
        """
        self.prices = prices 
        self.positions = positions
        self.pnl = None
        self.equity_curve = None
    
    def output_summary_stats(self, rf=.0, periods=252, compound=False, 
                             commission_fee=0):
        """Create a list of summary statistics for the portfolio including 
        Total Return, Sharpe Ratio, Max Drawdown, Drawdown Duration and Winning
        Rate.
        Args:
            rf: Risk free rate.
            periods: Number of days in a year.
            pdf: matplotlib pdf object.
        Returns:
            A summary statistics.
        """
        no_of_transactions, cf_vec = self._calc_transactions(commission_fee)
        self._calc_pnl(cf_vec)
        self._build_equity_curve(compound)

        total_return = self.equity_curve.iloc[-1]
        sharpe_ratio = self._calc_sharpe_ratio(rf, periods)
        max_dd = self._calc_max_drawdown()
        dd_duration = self._calc_max_drawdown_duration()
        winning_rate = self._calc_winning_rate()
        stats = [('Total Return', '{:0.2f}%'.format((total_return-1)*100)),
                 ('Sharpe Ratio', '{:0.4f}'.format(sharpe_ratio)),
                 ('Max Drawdown', '{:0.2f}%'.format(max_dd*100)),
                 ('Drawdown Duration', '{:d}'.format(dd_duration)), 
                 ('Winning Rate', '{:0.2f}%'.format(winning_rate*100)),
                 ('No of Transactions', '{:d}'.format(no_of_transactions))]
        return OrderedDict(stats)
    
    def _calc_pnl(self, commission_fee_vec):
        """Calculates strategy P&L based on prices and positions. 
        """
        self.pnl = self.prices.pct_change().multiply(self.positions.shift(1))
        self.pnl.iloc[0] = 0
        self.pnl -= commission_fee_vec
    
    def _build_equity_curve(self, compound):
        """Build the equity curve.
        """
        if compound:
            self.equity_curve = (self.pnl+1).cumprod()
        else:
            self.equity_curve = self.pnl.cumsum() + 1
        
    def _calc_sharpe_ratio(self, rf, periods):
        """Create the Sharpe ratio for the returns.
        Args:
            rf: Risk free rate.
            periods: Number of days in a year.
        Returns:
            A number represents Sharpe Ratio.
        """
        excess_rets = self.pnl - rf/periods
        return np.sqrt(periods)*np.mean(excess_rets)/np.std(excess_rets)
    
    def _calc_max_drawdown(self):
        """Calculate the max drawdown.
        Returns:
            A number represents maximum drawdown.
        """
        return 1-min(self.equity_curve/self.equity_curve.cummax())
    
    def _calc_max_drawdown_duration(self):
        """Calculate mdd duration.
        Returns:
            A number represents maximum drawdown duration.
        """
        dd = self.equity_curve-self.equity_curve.cummax()
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
    
    def _calc_winning_rate(self):
        """Calculate winning rates.
        Returns:
            A number represents winning rates.
        """
        #???? need to be refined????
        ud = self.prices.diff()
        ud[ud>0] = 1
        ud[ud<0] = -1
        return (ud==self.positions.shift(1)).sum()/(len(ud)-1)
    
    def _calc_transactions(self, commission_fee):
        """Calculate number of transactions and commission fees.
        Returns:
            A number represents the number of transactions.
        """
        transactions = self.positions.shift()!=self.positions
        no_of_transactions = transactions.sum()
        transactions_fee_vec = np.where(transactions, commission_fee, 0)
        return no_of_transactions, transactions_fee_vec
        
