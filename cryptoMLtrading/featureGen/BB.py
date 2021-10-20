#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import LoadDfs
import matplotlib.pyplot as plt


# In[2]:


#Retrieve df
#coins = LoadDfs.create_dataframes()
#df = coins["DOTUSDT"]


# In[3]:


#Calculate Technical indicator on df
def BB(df):
    period = 20
    # small time Moving average. calculate 20 moving average using Pandas over close price
    df['sma'] = df['close'].rolling(period).mean()
    # Get standard deviation
    df['std'] = df['close'].rolling(period).std()
    
    # Bollinger bands are normalised by dividing it by the closing price, that way the final value of these do not reflect 
    # the price directly and create a bias in the model.
    
    # Calculate Upper Bollinger band
    df['NormalisedUB'] = ( df['sma']  + (2 * df['std']) ) / df['close']
    # Calculate Lower Bollinger band
    df['NormalisedLB'] = ( df['sma']  - (2 * df['std']) ) / df['close']
        
    # Remove the std column
    df.drop(columns=["std", "sma"], inplace=True) 
    
    #Return df with new column of Technical Indicator
    return df


# In[4]:


#updated_df_BB = BB(df)
#updated_df_BB[-100:]


# In[5]:


#updated_df_BB = updated_df_BB[:10000] #(used to visualize the normalised BB with less data)


# In[ ]:


# Plotting it all together
#ax = updated_df_BB[['close', 'NormalisedLB', 'NormalisedUB']].plot(color=['blue', 'orange', 'yellow'])
#ax.fill_between(updated_df_BB.index, updated_df_BB['NormalisedLB'], updated_df_BB['NormalisedUB'], facecolor='orange', alpha=0.1)
#plt.show()


# In[ ]:





# In[ ]:




