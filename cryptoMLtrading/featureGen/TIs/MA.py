#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import numpy as np
import LoadDfs
import talib
import matplotlib.pyplot as plt


# In[61]:


#coins = LoadDfs.create_dataframes()
#dotusdt_df = coins["DOTUSDT"]


# In[55]:


def MA(dataframe):
    #Get MA through the closing price of the dataframe
    ma_periods = [9,20,30,50,200]
    ma_values = []
    for period in ma_periods:
        #change the period for each defined period
        ma_values.append(dataframe["close"].rolling(window=period).mean() / dataframe["close"])
    #Assign each value to the dataframe
    dataframe = dataframe.assign(MA9=ma_values[0])
    dataframe = dataframe.assign(MA20=ma_values[1])
    dataframe = dataframe.assign(MA30=ma_values[2])
    dataframe = dataframe.assign(MA50=ma_values[3])
    dataframe = dataframe.assign(MA200=ma_values[4])
    return dataframe


# In[59]:


#dotusdt_df_ma = MA(dotusdt_df)


# In[60]:


#Plot the MAs on a graph of 400 
#fig, ax = plt.subplots()
#ax.plot(dotusdt_df_ma["close"][1000:1400],color='blue', label = "price")
#ax.plot(dotusdt_df_ma["MA9"][1000:1400],color="red",label = "ma9")
#ax.plot(dotusdt_df_ma["MA20"][1000:1400],color="orange",label = "ma20")
#ax.plot(dotusdt_df_ma["MA30"][1000:1400],color="green", label = "ma30")
#ax.plot(dotusdt_df_ma["MA50"][1000:1400],color = "black",label = "ma50")
#ax.plot(dotusdt_df_ma["MA200"][1000:1400],color = "yellow",label = "ma200")
#ax.legend(loc = 'lower right')
#fig = plt.gcf()
#fig.set_size_inches(18.5, 10.5)
#ax.set_axisbelow(True)
#ax.yaxis.grid(color='gray', linestyle='dashed')
#plt.ylabel("price")
#plt.xlabel("time")
#plt.show()


# In[ ]:




