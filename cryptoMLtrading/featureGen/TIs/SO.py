#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import LoadDfs
import matplotlib.pyplot as plt
#!pip install pandas_ta
import pandas_ta as ta


#Calculate Technical indicator on df
def SO(df):
    # # Calculate Stochastic Oscillator values using the pandas_ta library
    df.ta.stoch(high='high', low='low', k=14, d=3, append=True)

    #Return df with new column of Technical Indicator
    return df


#################### Use case example with graph included ####################

#Retrieve df
#coins = LoadDfs.create_dataframes()
#df = coins["DOTUSDT"]

#updated_df_SO = MACD(df)
#updated_df_SO[-100:]


#test_df_SO = updated_df_SO[-1000:] #(used to visualize the SO with less data)

# Plotting it all together
#ax = test_df_SO[['close', 'STOCHk_14_3_3', 'STOCHd_14_3_3']].plot(color=['blue', 'yellow', 'green'])
#plt.show()




