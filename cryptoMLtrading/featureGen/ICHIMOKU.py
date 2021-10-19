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


# In[60]:


#Calculate Technical indicator on df
def ICHIMOKU(df):
    # Calculate Hichimoku values using the pandas_ta library
    df.ta.ichimoku(high='high', low='low', window1=9, window2=26, window3=52, append=True)
    # Normalise Hichimoku values
    df['ISA_9'] = df['ISA_9'] / df['close']
    df['ISB_26'] = df['ISB_26'] / df['close']
    df['ITS_9'] = df['ITS_9'] / df['close']
    df['IKS_26'] = df['IKS_26'] / df['close']
    df['ICS_26'] = df['ICS_26'] / df['close']

    #Return df with new column of Technical Indicator
    return df


# In[61]:


#updated_df_ICHIMOKU = ICHIMOKU(df)
#updated_df_ICHIMOKU.head(-100)


# In[63]:


#test_df_ICHIMOKU = updated_df_ICHIMOKU[-100:] #(used to visualize the ICHIMOKU with less data)


# In[64]:


# Plotting it all together
#ax = test_df_ICHIMOKU[['ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26']].plot(color=['orange', 'yellow', 'red', 'green', 'purple'])
#ax.fill_between(test_df_ICHIMOKU.index, test_df_ICHIMOKU['ISA_9'], test_df_ICHIMOKU['ISB_26'], facecolor='red', alpha=0.3)
#plt.show()


# In[ ]:




