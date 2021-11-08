#!/usr/bin/env python
# coding: utf-8

# In[1]:


import websocket, json, pprint
import numpy as np
import config
from binance.client import Client
from binance.enums import *
import datetime
import asyncio
import threading
import pandas as pd
import pickle
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('../models')
sys.path.append('../featureGen')


#import TI
from RSI import RSI
from SO import SO
from MA import MA
from BB import BB
from MACD import MACD
from FR import FR
import random
from collections import deque


# In[1]:


#Load candles of DOTUSDT AND SOLUSDT
dotusdt_stream = "dotusdt_perpetual@continuousKline_5m"
solusdt_stream = "solusdt_perpetual@continuousKline_5m"
SOCKET1 = f"wss://fstream.binance.com/stream?streams={solusdt_stream}/{dotusdt_stream}"


# In[5]:


solusdt = pd.DataFrame(columns = ["open", "high", "low", "close", "volume", "num_of_trades"])
dotusdt = pd.DataFrame(columns = ["open", "high", "low", "close", "volume", "num_of_trades"])

dataframes = {"solusdt":solusdt, "dotusdt": dotusdt}
dataframes_TI =  {"solusdt":None, "dotusdt": None}

ensemble_predictions = None


# In[6]:


def make_cols(test_df):
    #Make a dictionary of dataframe:column names, e.g for df2, we need high2,low2,vol2
    og_columns = test_df.columns[:-1]
    
    periods = 4
    column_names={}
    for i in range(periods):
        columns = []
        for name in og_columns:
            columns.append(name+str(i+2))
        column_names[i+1] = columns
    return column_names, og_columns

test_df = pd.read_csv('../featureGen/PROCESSED_COINS/Sequenced/DF_sequence_DOTUSDT.csv')
column_names, og_columns = make_cols(test_df)


# In[7]:


client = Client(config.API_KEY, config.API_SECRET)


# In[8]:


def add_dataframe(coin , json_message):
    global dataframes
    
    opt = json_message["data"]["k"]
    dataframes[coin] = dataframes[coin].append({"open":float(opt["o"]), "high":float(opt["h"]), 
                                                "low":float(opt["l"]), "close":float(opt["c"]),
                                                "volume":float(opt["v"]), "num_of_trades":float(opt["n"])},
                                               ignore_index = True)


# In[9]:


def df_for_models_TI(dataframe):
    #Append tech indicators to Original dataframe
    full_df_rsi = RSI(dataframe)
    full_df_bb = BB(full_df_rsi)
    full_df_SO = SO(full_df_bb)
    full_df_MA = MA(full_df_SO)
    full_df_MACD = MACD(full_df_MA)
    full_df = FR(full_df_MACD)

    return full_df


# In[10]:


def process_df(df):
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
    df.drop(columns=['open','close'], inplace=True)
    
    return df


# In[11]:


def concat(full_df):
    global column_names, og_columns
    
    concat_dataframe = full_df.copy()
    dfs = {}
    for i in range(1,5):
        dfs[i] = full_df.shift(i).copy()

    concat_dataframe.reset_index(inplace = True, drop = True)
    concat_dataframe.reset_index(inplace = True)
    concat_dataframe.set_index('index',inplace=True)    

    for i in range(1,5):
        cols = dict(zip(og_columns, column_names[i]))
        dfs[i] = dfs[i].rename(columns = cols)
        dfs[i].reset_index(inplace = True, drop = True)
        dfs[i].reset_index(inplace = True)
        dfs[i].set_index('index',inplace = True)
        concat_dataframe = concat_dataframe.join(dfs[i])
        
    return concat_dataframe


# In[12]:


# Loading models
RF = pickle.load(open('../models/trained_models_2/RF_model_1_EE.sav', 'rb'))
LR = pickle.load(open('../models/trained_models_2/LR_model_1_EE.sav', 'rb'))
NN1 = keras.models.load_model('../models/trained_models_2/NN_model_1_All_coins_EE')
NN2 = keras.models.load_model('../models/trained_models_2/NN_model_2_All_coins_EE')
RNN = keras.models.load_model('../models/trained_models_2/RNN_model_1_ALL_COINS_EE')
ensemble = keras.models.load_model("../models/trained_models_2/ensemble_1")


# In[13]:


def create_seq(dataframe):
    sequential_data = []
    prev_days = deque(maxlen=5)
    for i in dataframe.values:
        prev_days.append([n for n in i])
        if len(prev_days) == 5:
            sequential_data.append(np.array(prev_days))
    return np.array(sequential_data)


# In[45]:


def predict(concat_df_values, sequenced_values):
    RF_preds = RF.predict_proba(concat_df_values)
    LR_preds = LR.predict_proba(concat_df_values)
    NN1_preds = NN1.predict(concat_df_values)
    NN2_preds = NN2.predict(concat_df_values)
    RNN_preds = RNN.predict(sequenced_values)

    concat_preds_EE = np.concatenate((RF_preds,LR_preds,NN1_preds,NN2_preds,RNN_preds), axis=1)
    ensemble_preds = ensemble.predict(concat_preds_EE)

    return ensemble_preds


# In[15]:


def logic():
    return None
def execute_trade():
    return None


# In[16]:


def on_open(ws):
    print('opened connection')

def on_close(ws):
    print('closed connection')
    
def on_error(ws, error):
    print(error)

def on_message(ws, message):
    
    global dataframes, dataframes_TI, dataframes_, ensemble_prediction
    
    #LOADS JSON MESSAGE (PINGS APPROX 3 A SECOND)
    json_message = json.loads(message)

    #If Ping returns data displaying end of candle, follow.
    if json_message["data"]["k"]["x"] == False: 
            coin = json_message["data"]["ps"].lower()
    
            add_dataframe(coin, json_message)
        
            # **Insert Method **
            #----- Perform analysis on the dataframe (MIN 250 ROWS)
            #----------E.g RSI, FR, MACD....
            #----------Return dataframe with TI  
            
            # Minimum 250 rows of data are previously in order to compute some TIs
            #if len(dataframes[coin]) % 100 == 0:
                #print(len(dataframes[coin]))
            if len(dataframes[coin]) >= 300:
                dataframes_TI[coin] = df_for_models_TI(dataframes[coin][-300:])
                
                #Data Normalisation
                dataframes_TI[coin] = process_df(dataframes_TI[coin])
  
                concat_dataframe = concat(dataframes_TI[coin])
                RNN_prepared = create_seq(dataframes_TI[coin]
                
                ensemble_prediction = predict(concat_dataframe[-1].values, RNN_prepared[-1])
                                          
                # ** Insert Method **
                #------ LOGIC METHOD, depending on parameters we set
                #----------Return Long, Short or None

                # ** Insert Method **
                #------ Execute trades

        
        
    # reduce amount of rows in dataframe
        
    #------ if during a 5 minute period (not open or close) the data obtained does not respect the  
    #------ sequential time stamp order (e.g. due to connection failure)
    #------ another variable keeping track of the time stamps of all the data required should indicate (doing the maths)
    #------ if any 5 minute slot of data has been missed. 
    #------ if this is the case, the missing time stamps should be used to request to data for that specific date
    #------ and fill in the dataframes with the corresponding data in their corresponding indexes
    #elif json_message["data"]["k"]["x"] == False:

    
ws = websocket.WebSocketApp(SOCKET1, on_open=on_open, on_close=on_close, on_message=on_message, on_error = on_error)
ws.run_forever()


# In[ ]:




