#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[19]:


class run_concat_sequenced():
    
    def __init__(self, dataframes_dict,stop_gain,stop_loss):
        self.coins = dataframes_dict
        self.stop_gain = stop_gain
        self.stop_loss = stop_loss
        self.run_file()
        
    def run_file(self):
        types = ["concat","sequenced"]
        for type_ in types:
            for key,value in self.coins.items():
                dataframe = self.coins[key]
                print(key)
                full_df_RSI = RSI(dataframe)
                full_df_BB = BB(full_df_RSI)
                full_df_SO = SO(full_df_BB)
                full_df_MA = MA(full_df_SO)
                full_df_MACD = MACD(full_df_MA)
                full_df = FR(full_df_MACD)

                full_df_labels = LABEL(full_df,self.stop_gain,self.stop_loss)
                full_df_complete = CLEAN(full_df_labels)
                look_back = 5
                
                if type_ == "concat":
                    full_df_concat = self.append(full_df_complete, look_back)
                    NAME = F"Strategy2/DF_{look_back}_Candles_Concat_{key}.csv"
                    full_df_concat.to_csv(f"../featureGen/PROCESSED_COINS/{NAME}",index = False) 
                
                elif type_ == "sequenced":
                    
                    NAME = F"Strategy2/DF_sequence_{key}.csv"
                    full_df_complete.to_csv(f"../featureGen/PROCESSED_COINS/{NAME}",index = False)

                
    def append(self, data, look_back = 5,columns_size = 20):

        column_og = data.columns

        periods = 4
        column_names={}
        for i in range(periods):
            columns = []
            for name in data.columns:
                columns.append(name+str(i+2))
            column_names[i+2] = columns

        Nan_list = ["NaN"]*columns_size
        Nan_df = pd.DataFrame([Nan_list], columns = data.columns)

        dfs = {}
        look_back = 5
        for i in range(2,look_back+1):
            #Append Nan rows to the first i lines of the dataframe, depending on how far looking back
            dfs["dataframe"+str(i)] = Nan_df
            for j in range(i-2):
                dfs["dataframe"+str(i)] = dfs["dataframe"+str(i)].append(Nan_df)

        #append the data to each Nan dataframe, e.g Dataframe5 with 5 nan rows will add all the original data to it,
        #but it wil start from row 5 downwards
        for i in range(2,look_back+1):
            dfs["dataframe"+str(i)] = dfs["dataframe"+str(i)].append(data)

        cols = {}
        #With the column names of the OG data, append a number to each column depending on the dataframe
        #i.e dataframe5 will have "High5, low5, volume5"
        for i in range(2,look_back+1):
            for j in range(len(column_names[i])):
                a = list(dfs["dataframe"+str(i)].columns)
                cols[a[j]] = column_names[i][j]
            #Reset indexes so its 0-n and create a main index which is used to join tables
            dfs["dataframe"+str(i)] = dfs["dataframe"+str(i)].rename(cols, axis=1)
            dfs["dataframe"+str(i)].reset_index(inplace = True, drop = True)
            dfs["dataframe"+str(i)].reset_index(inplace = True)
            dfs["dataframe"+str(i)].set_index('index',inplace =True)

        #Resetting the original dataframe index so it can match with newly created tables when indexing
        data.reset_index(inplace = True, drop = True)
        data.reset_index(inplace = True)
        data.set_index('index',inplace=True)

        #Joining the Dataframes together
        for i in range(2,look_back+1):
            #print(f"Candle N-{i} added")
            data = data.join(dfs["dataframe"+str(i)])

        #removing the labels in each column apart from OG label
        for i in range(2,look_back+1):
            del data["label"+str(i)]

        #Popping the label column and ammending the label column to the end of the dataframe
        label = data.pop("label")
        data["label"] =  label
        return data[look_back:]

