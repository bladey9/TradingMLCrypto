#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import LoadDfs
import talib
import matplotlib.pyplot as plt

def FR(dataframe):
    #create the lists for FR grade and FR percent
    fib_grades = []
    FR_percent = []
    for i in range(0,len(dataframe)):
        #append None to the first 250 lists
        if i <250:
            fib_grades.append(None)
            FR_percent.append(None)
        else:
            #find max and min values of the last 250 rows
            maX = dataframe["close"][i-250:i+1].max()
            miN = dataframe["close"][i-250:i+1].min()

            #calc the fibonacci levels
            diff = maX - miN
            level1 = maX - diff * 0.236
            level2 = maX - diff * 0.382
            level3 = maX - diff * 0.5
            level4 = maX - diff * 0.618
            curr_price = dataframe["close"][i]

            #calc fib grade and fib percent inside the grade
            if curr_price  >= level1:
                fib_grade = 0
                height_fibperiod = ((curr_price - level1) / (maX - level1))
            elif curr_price  >= level2:
                fib_grade = 1
                height_fibperiod = ((curr_price - level2) / (level1 - level2))
            elif curr_price  >= level3:
                fib_grade = 2
                height_fibperiod = ((curr_price - level3) / (level2 - level3))
            elif curr_price  >= level4:
                fib_grade = 3
                height_fibperiod = ((curr_price - level4) / (level3 - level4))
            elif curr_price  < level4:
                fib_grade = 4
                height_fibperiod = ((curr_price - miN) / (level4 - miN))

            fib_grades.append(fib_grade)
            FR_percent.append(height_fibperiod)
    #assignt the lists
    dataframe = dataframe.assign(FR_grade=fib_grades)
    dataframe = dataframe.assign(FR_percent=FR_percent)
    return dataframe





