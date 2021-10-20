#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
pd.options.display.max_columns= None
import numpy as np
import LoadDfs
import talib
import matplotlib.pyplot as plt

#import TI
from RSI import RSI
from SO import SO
from MA import MA
from BB import BB
from MACD import MACD
from ICHIMOKU import ICHIMOKU
from FR import FR

#clean and append labels
from CLEAN import CLEAN
from Label import LABEL


# In[2]:


#Load original dataframe
coins = LoadDfs.create_dataframes()
dotusdt_df = coins["DOTUSDT"] 


# In[3]:


#Append tech indicators to Original dataframe

full_df_rsi = RSI(dotusdt_df)
print("rsi completed")
full_df_bb = BB(full_df_rsi)
print("bb completed")
full_df_SO = SO(full_df_bb)
print("SO completed")
full_df_MA = MA(full_df_SO)
print("MA completed")
full_df_MACD = MACD(full_df_MA)
print("MACD completed")
full_df_ICHIMOKU = ICHIMOKU(full_df_MACD)
print("ICHIMOKU completed")
full_df = FR(full_df_ICHIMOKU)
print("FR completed")


# In[ ]:


#save file as dataframe with tech indicators


# In[5]:


#Appennds labels to it
full_df_labels = LABEL(full_df)
full_df_complete = CLEAN(full_df_labels)


# In[9]:


full_df_complete.to_csv("df_for_model_1.0",index=False)


# In[ ]:




