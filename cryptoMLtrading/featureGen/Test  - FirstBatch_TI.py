#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
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


# 
# **Following line saves it as a csv format so you can load it in another notebook. We do our work there and then we talk about how we are going to append it to this file.**

# In[11]:


#save file as dataframe with tech indicators
full_df.to_csv("df_with_TI",index=False)


# In[ ]:




