# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:31:45 2018

@author: rl17174
"""
import numpy as np

class Backtest(object):
    """Represents a backtest.
    """
    
    def __init__(self, prices, positions):
        self.prices = prices 
        self.positions = positions
        self.pnl = None
        self.equity_curve = None
        
    def _calc_pnl(self):
        """Calculates strategy pnl based on prices and positions. 
        """
        self.pnl = self.prices.pct_change().multiply(self.positions.shift(1))
        self.pnl.iloc[0] = 0
    
    def _build_equity_curve(self):
        """Build the equity curve.
        """
        self.equity_curve = (self.pnl+1).cumprod()
    
    def _winning_rate(self):
        """Calculates winning rates.
        """
        #???? need to be refined????
        ud = self.prices.diff()
        ud[ud>0] = 1
        ud[ud<0] = -1
        return (ud==self.positions.shift(1)).sum()/(len(ud)-1)
    
    def _calc_max_drawdown_duration(self):
        """Calculate mdd duration.
        """
        roll_max = np.maximum.accumulate(self.equity_curve)
        duration = self.equity_curve-roll_max
        duration.index = range(len(duration))
        idx = np.where(duration==.0)
        return np.diff(idx).max()-1
              
    def _calc_max_drawdown(self):
        """Calculates the max drawdown of a price series. If you want the
        actual drawdown series, please use to_drawdown_series.
        """
        return 1-(self.equity_curve/self.equity_curve.expanding(min_periods=1)\
                  .max()).min()
          
    def _calc_sharpe_ratio(self, rf, periods):
        """Create the Sharpe ratio for the returns. 
        """
        excess_rets = self.pnl - rf/periods
        return np.sqrt(periods)*np.mean(excess_rets)/np.std(excess_rets)

    def output_summary_stats(self, rf=.0, periods=252):
        """Creates a list of summary statistics for the portfolio such
        as Sharpe Ratio and drawdown information.
        """
        self._calc_pnl()
        self._build_equity_curve()
        self.equity_curve.plot()
        total_return = self.equity_curve.iloc[-1]
        sharpe_ratio = self._calc_sharpe_ratio(rf, periods)
        max_dd = self._calc_max_drawdown()
        dd_duration = self._calc_max_drawdown_duration()
        winning_rate = self._winning_rate()
        stats = [("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
                 ("Sharpe Ratio", "%0.4f" % sharpe_ratio),
                 ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
                 ("Drawdown Duration", "%d" % dd_duration), 
                 ("Winning Rate", "%0.2f%%" % (winning_rate * 100.0))]
        return stats
