#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import LoadDfs
import matplotlib.pyplot as plt
#!pip install pandas_ta
import pandas_ta as ta


# In[2]:


#Retrieve df
#coins = LoadDfs.create_dataframes()
#df = coins["DOTUSDT"]


# The MACD represents 3 district values, each of which are interconnected. The insights provided by the MACD require one to understand how each of these values is calculated, what they represent, and the implications of movement relative to one another. Below are the MACD’s main primary signals.
# 
# MACD – the value of an exponential moving average (EMA) subtracted from another EMA with a shorter lookback period. Common values are 26 days for the longer EMA and 12 for the shorter. This is referred to as the ‘slow’ signal line of the MACD.
# 
# Signal– the EMA of the MACD of a period shorter than the shortest period used in calculating the MACD. Typically, a 9-day period is used. This is referred to as the ‘fast’ signal line.
# 
# Difference – The difference between the MACD – Trigger line is used to represent current selling pressures in the marketplace. This value is commonly charted as a histogram overlaying the MACD + Trigger signals. A positive value for this signal represents a bullish trend whereas a negative value indicates a bearish one.

# In[3]:


#Calculate Technical indicator on df
def MACD(df):

    # # Calculate MACD values using the pandas_ta library
    df.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
    #Return df with new column of Technical Indicator
    return df


# In[4]:


#updated_df_MACD = MACD(df)
#updated_df_MACD[-100:]


# In[5]:


#test_df_MACD = updated_df_MACD[:5000] #(used to visualize the MACD with less data)


# In[7]:


# Plotting it all together
#ax = test_df_MACD[['close', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']].plot(color=['blue', 'red', 'yellow', 'green'])
#plt.show()


# In[ ]:





# In[ ]:




