#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler


# In[ ]:


#df = pd.read_csv('df_with_TI')
#df[-5:]


# In[2]:


def CLEAN(df):
    # Normalise high and low prices, and RSI
    df['high'] = df['high'] / df['close']
    df['low'] = df['low'] / df['close']
    df['RSI'] = df['RSI'] / 100
    
    # Scaling volume and number of trades
    scaler = StandardScaler()
    sca_vol = scaler.fit_transform(df[["volume"]])
    df["volume"] = sca_vol
    sca_NoT = scaler.fit_transform(df[["num_of_trades"]])
    df["num_of_trades"] = sca_NoT

    # Remove columns
    df.drop(columns=['open','close', 'open_time','close_time','ignore', 'quote_asset_vol','taker_buy_base_asset_vol','taker_buy_quote_asset_vol'], inplace=True)
    df.dropna(inplace=True)
    
    check = True
    current = df.index[0] +1
    for index, row in df.iterrows():
        if index+1 != current:
            check = False
        current +=1
    
    if check:
        return df 
    else:
        return 'ERROR: Index Missmatch'


# In[ ]:


#df_clean = CLEAN(df)
#df_clean[-5:]


# In[ ]:


#df_clean.shape


# In[ ]:




