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


# In[3]:


#Calculate Technical indicator on df
def SO(df):
    # # Calculate Stochastic Oscillator values using the pandas_ta library
    df.ta.stoch(high='high', low='low', k=14, d=3, append=True)

    #Return df with new column of Technical Indicator
    return df


# In[4]:


#updated_df_SO = MACD(df)
#updated_df_SO.head(-100)


# In[23]:


#test_df_SO = updated_df_SO[-1000:] #(used to visualize the SO with less data)


# In[24]:


# Plotting it all together
#ax = test_df_SO[['close', 'STOCHk_14_3_3', 'STOCHd_14_3_3']].plot(color=['blue', 'yellow', 'green'])
#plt.show()


# In[ ]:





# In[ ]:




