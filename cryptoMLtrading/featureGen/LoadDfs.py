#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from datetime import datetime


# In[ ]:


def create_dataframes():
    main_path = 'originalData/data/futures/um/monthly/klines'
    coins = [f for f in listdir(main_path) if 'USDT' in f]
    coin_5m = {}
    for coin in coins:
        path_5m = main_path + '/' + coin + '/5m/'
        loDf = [] # list of dataframes to afterwards append them all in a single one
        for f in listdir(path_5m): # loop through every file in the 'path_5m' directory
            if isfile(join(path_5m, f)):
                single_df = pd.read_csv(os.path.join(path_5m,f))
                single_df.columns = ["open_time", "open", "high","low","close","volume","close_time","quote_asset_vol","num_of_trades","taker_buy_base_asset_vol","taker_buy_quote_asset_vol","ignore"]
                loDf.append(single_df)
        
        concat_dfs = pd.concat(loDf).reset_index() # concatenate all df into a single one for each individual coin 
        for index,row in concat_dfs.iterrows():
            concat_dfs.at[index,"open_time"] = datetime.fromtimestamp(int(concat_dfs["open_time"][index])/1000)
        concat_dfs.sort_values(by=['open_time'], inplace=True, ascending=True)
        del concat_dfs['index'] # deleting named 'index'
        coin_5m[coin] = concat_dfs

    return coin_5m
        


# In[ ]:


#coin_5m = create_dataframes()


# In[ ]:


#coin_5m['DOTUSDT'].head()


# In[ ]:




