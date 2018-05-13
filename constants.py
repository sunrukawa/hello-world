#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 15:27:29 2018

@author: sunrukawa
"""

# note: since the night of 05/03/2016, the trading hours for 'rb' and 'bu' 
# changed from '1:00:00 am' to '23:00:00 pm'.
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

# note: only per-side commission fee charged by Shanghai future exchange for 
# 'ag', 'bu', 'rb' and 'zn', while round-turn for 'ru'.
# the per-side fee for 'zn' is 3 CNY for a contract, which each contract worths
# roughly 13,000*5~21,000*5, so the upper bound would be 3/13,000/5.
commission_fees = {'ag': 0.00005,
                   'bu': 0.00003,
                   'rb': 0.000045,
                   'ru': 0.000045*2,
                   'zn': 0.00005}
                      