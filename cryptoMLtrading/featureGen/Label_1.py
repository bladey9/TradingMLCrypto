#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[21]:


#Assign label to df
#Label 1 = Reached high without reaching low
#Label 2 = Reached neither high or low
#Label 3 = Reached high and low
#Label 4 = Reached low without reaching high

def LABEL_1(dataframe):
    for index,row in dataframe.iterrows():
        increase = (dataframe["high"][index] / dataframe["open"][index]) - 1
        decrease = (1 - (dataframe["low"][index]/dataframe["open"][index]))
        
       
        dataframe.at[index-1,"label"] = 0
        
        if increase >= 0.005 and decrease <= 0.002:
            dataframe.at[index-1,"label"] = 1
            
        if increase <= 0.002 and decrease >= 0.005:
            dataframe.at[index-1,"label"] = 2
            
        if increase >= 0.005 and decrease >= 0.005:
            dataframe.at[index-1,"label"] = 3
            
            
    dataframe.drop(dataframe.tail(2).index,inplace=True) 
    
    return dataframe


# In[ ]:




