#!/usr/bin/env python
# coding: utf-8

# In[1]:


import websocket, json, pprint, talib, numpy
import config
from binance.client import Client
from binance.enums import *
import datetime
import asyncio
import threading


# In[2]:


import pandas as pd


# In[13]:


#SOCKET = "wss://stream.binance.com:9443/ws/dotusdt@kline_5m"
#SOCKET = "wss://fstream.binance.com/ws/dotusdt@kline_1m"

dotusdt = "dotusdt_perpetual@continuousKline_1m"
solusdt = "solusdt_perpetual@continuousKline_1m"
SOCKET1 = f"wss://fstream.binance.com/stream?streams={solusdt}/{dotusdt}"


# In[14]:


solusdt = pd.DataFrame(columns = ["open", "high", "low", "close", "volume", "num_of_trades"])
dotusdt = pd.DataFrame(columns = ["open", "high", "low", "close", "volume", "num_of_trades"])


# In[16]:


client = Client(config.API_KEY, config.API_SECRET)


# In[24]:


#info = client.get_symbol_info('BNBBTC')

def add_dataframe(dataframe , json_message):
    global solusdt, dotusdt
    
    if json_message['stream'] == f'{dataframe}@kline_1m':
            #Get price data
            opt = json_message["data"]["k"]
            #Append certain data to SOL DF
            print("hey")
            dataframe = dataframe.append({"open":opt["o"], "high":opt["h"], "low":opt["l"], "close":opt["c"],
                                      "volume":opt["v"], "num_of_trades":opt["n"]},ignore_index = True)
        
    


# In[25]:


def on_open(ws):
    print('opened connection')

def on_close(ws):
    print('closed connection')
    
def on_error(ws, error):
    print(error)

def on_message(ws, message):
    
    global solusdt, dotusdt
    
    #LOADS JSON MESSAGE (PINGS APPROX 3 A SECOND)
    json_message = json.loads(message)

    #If Ping returns data displaying end of candle, follow.
    if json_message["data"]["k"]["x"] == True: 
        
        if json_message['stream'] == "solusdt_perpetual@continuousKline_1m":
            add_dataframe(solusdt, json_message)
        
            # **Insert Method **
            #----- Perform analysis on the dataframe (MIN 250 ROWS)
            #----------E.g RSI, FR, MACD....
            #----------Return dataframe with TI

            # **Insert Method **
            #------ Concat dataframe and also have a sequenced dataframe (requires no change from OG dataframe)
            #----------Return dataframes for 1.ALL Models exc RNN, 2. RNN model

            # **Insert Method **
            #------ Data Refactoring for NN input and RNN input (Sequence)
            #----------Return data refactoring 

            # ** Insert Method **
            #------ Run models on rows
            #----------Return predictions

            # ** Insert Method **
            #------ LOGIC METHOD, depending on parameters we set
            #----------Return Long, Short or None

            # ** Insert Method **
            #------ Execute trades

    
ws = websocket.WebSocketApp(SOCKET1, on_open=on_open, on_close=on_close, on_message=on_message, on_error = on_error)
ws.run_forever()


# In[ ]:


#ws = websocket.WebSocketApp(SOCKET1, on_open=on_open, on_close=on_close, on_message=on_message, on_error = on_error)
#ws.run_forever()

#wst = threading.Thread(target = ws.run_forever)
#wst.start()

#code here runs on separate thread only ONCE
#Cleaning or procedures here. 

#wst.close()


# - A SINGLE CONNECTION IS ONLY VALID FOR 24 HOURS - EXPECT TO BE DISCONNECTED AT THE 24 HOUR MARK
# - THE WB SERVER WILL SEND A PING FRAME EVERY 5MINS. IF THE WB SERVER DOES NOT RECEIVE A PONG FRAME BACK
# FROM THE CONNECTION WITHIN A 15MIN PERIOD, THE CONNECTION WILL BE DISCONNECTED
# - SOME COINS ALLOW STOP LIMITS ON SPOT, MARGIN, FUTURES

# In[ ]:




