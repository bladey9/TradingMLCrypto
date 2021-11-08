#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import LoadDfs
import talib
import matplotlib.pyplot as plt


def RSI(dataframe):
    #Get closing price
    closing_prices = dataframe["close"]
    #Retrieve RSI values
    rsi = talib.RSI(closing_prices)
    #assign rsi values to df
    dataframe = dataframe.assign(RSI=rsi)
    return dataframe
        


#################### Use case example with graph included ####################

#retrieve Dataframe
#coins = LoadDfs.create_dataframes()
#dotusdt_df = coins["DOTUSDT"] 

#Append RSI column to original Dataframe
#df = RSI(dotusdt_df)

#Plot the values
#ax = df[['RSI']].plot(color=['blue'])
#ax.fill_between(updated_df_BB.index, updated_df_BB['lowerBollinger'], updated_df_BB['upperBollinger'], facecolor='orange', alpha=0.1)
#plt.show()





