#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[6]:


#Assign label to df
#Label 1 = Reached high without reaching low
#Label 2 = Reached neither high or low
#Label 3 = Reached high and low
#Label 4 = Reached low without reaching high

def LABEL(dataframe):
    for index,row in dataframe.iterrows():
        increase = (dataframe["high"][index] / dataframe["open"][index]) - 1
        decrease = (1 - (dataframe["low"][index]/dataframe["open"][index]))
        if increase > 0.005 and decrease < 0.003:
            dataframe.at[index-1,"label"] = 1
        elif increase < 0.003 and decrease > 0.005:
            dataframe.at[index-1,"label"] = 4
        elif increase > 0.003 and decrease > 0.003:
            dataframe.at[index-1,"label"] = 3
        else:
            dataframe.at[index-1,"label"] = 2
            
    dataframe.drop(dataframe.tail(2).index,inplace=True) 
    
    return dataframe


# In[ ]:




