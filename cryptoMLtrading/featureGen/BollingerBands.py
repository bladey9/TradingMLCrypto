#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import LoadDfs
import matplotlib.pyplot as plt


# In[2]:


#Retrieve df
coins = LoadDfs.create_dataframes()
dotusdt_df = coins["DOTUSDT"]


# In[6]:


#Calculate Technical indicator on df
def BB():
    period = 20
    # small time Moving average. calculate 20 moving average using Pandas over close price
    dotusdt_df['sma'] = dotusdt_df['close'].rolling(period).mean()
    # Get standard deviation
    dotusdt_df['std'] = dotusdt_df['close'].rolling(period).std()
    # Calculate Upper Bollinger band
    dotusdt_df['upperBollinger'] = dotusdt_df['sma']  + (2 * dotusdt_df['std'])
    # Calculate Lower Bollinger band
    dotusdt_df['lowerBollinger'] = dotusdt_df['sma']  - (2 * dotusdt_df['std'])
    # Remove the std column
    dotusdt_df.drop(columns="std", inplace=True) 
    
    #Return df with new column of Technical Indicator
    return dotusdt_df


# In[7]:


#updated_df_BB = BB()


# In[20]:


#updated_df_BB = updated_df_BB[:100] (used to visualize the BB with less data)


# In[21]:


# Plotting it all together
#ax = updated_df_BB[['close', 'lowerBollinger', 'upperBollinger']].plot(color=['blue', 'orange', 'yellow'])
#ax.fill_between(updated_df_BB.index, updated_df_BB['lowerBollinger'], updated_df_BB['upperBollinger'], facecolor='orange', alpha=0.1)
#plt.show()


# In[ ]:




