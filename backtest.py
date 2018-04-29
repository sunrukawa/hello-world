# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:31:45 2018

@author: rl17174
"""

import numpy as np
import matplotlib.pyplot as plt

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
    
    def output_summary_stats(self, multi_pdf, rf=.0, periods=252):
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
        self._calc_pnl()
        self._build_equity_curve()
        self.equity_curve.plot(figsize=(20,15), fontsize=15)
        multi_pdf.savefig()
        plt.clf()
        total_return = self.equity_curve.iloc[-1]
        sharpe_ratio = self._calc_sharpe_ratio(rf, periods)
        max_dd = self._calc_max_drawdown()
        dd_duration = self._calc_max_drawdown_duration()
        winning_rate = self._winning_rate()
        stats = [("Total Return", "%0.2f%%" % ((total_return-1)*100)),
                 ("Sharpe Ratio", "%0.4f" % sharpe_ratio),
                 ("Max Drawdown", "%0.2f%%" % (max_dd*100)),
                 ("Drawdown Duration", "%d" % dd_duration), 
                 ("Winning Rate", "%0.2f%%" % (winning_rate*100))]
        return stats
    
    def _calc_pnl(self):
        """Calculates strategy P&L based on prices and positions. 
        """
        self.pnl = self.prices.pct_change().multiply(self.positions.shift(1))
        self.pnl.iloc[0] = 0
    
    def _build_equity_curve(self):
        """Build the equity curve.
        """
        self.equity_curve = (self.pnl+1).cumprod()
        
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
        return 1-(self.equity_curve/self.equity_curve.expanding(min_periods=1)
                  .max()).min()
    
    def _calc_max_drawdown_duration(self):
        """Calculate mdd duration.
        Returns:
            A number represents maximum drawdown duration.
        """
        roll_max = np.maximum.accumulate(self.equity_curve)
        duration = self.equity_curve-roll_max
        duration.index = range(len(duration))
        idx = np.where(duration==.0)
        return np.diff(idx).max()-1
    
    def _winning_rate(self):
        """Calculate winning rates.
        Returns:
            A number represents winning rates.
        """
        #???? need to be refined????
        ud = self.prices.diff()
        ud[ud>0] = 1
        ud[ud<0] = -1
        return (ud==self.positions.shift(1)).sum()/(len(ud)-1)
