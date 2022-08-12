# TradingMLCrypto

NOTE - Only CEX trading bot is shown. Due to multiple authors of the DEX code, unable to share.

--ML--

A Project in finding potential trading strategies for intraday trading using Machine learning.

In order to download the required data for training and testing this project, the following shell code must be executed:

`python cryptoMLtrading/originalData/trading.py -s SOLUSDT FTMUSDT DOTUSDT MATICUSDT ICPUSDT OMGUSDT -i 1m 5m 15m -y 2020 2021 -t um -c 1`

The Main Objective of this project is to find a suitable trading strategy through the following:

- Obtain and extract historical cryptocurrency data from the official binance platform
- Preprocess the data in order to extract relevant information (open,high,low,close,volume,num_of_trades etc)
- Perform technical analysis on the data
- Use machine learning architectures to train models (RF, NN, RNN and LR)
- Gather the predictions into an ensemble model to finalise a probability prediction for a potential strategy on x minute candles.
- Connect to the Binance API to execute trades depending on the strategy used


**Main folder:**

Binance Connection

- Used to retrieve a websocket connection for live Binance data
- Run a potential strategy on the live data

Pancakeswap Connection

**models**

Pipeline

- Used to create and test new potential strategies depending on the label given (creates new dataframes). 

Model Classes

- Machine Learning architectures

**featureGen**

Files here support the generation of feature preprocessing and different trading strategies

Dataframes of all coins also processed coins

PREPROCESSED_COINS

- Concat = last 5 candles amended into one row
- Sequenced = Normal row by row dataframe representing sequential candles

TIs

- Files here used to run technical indicators on dataframe
