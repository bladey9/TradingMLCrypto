#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os

import sys
sys.path.append("../featureGen/")
sys.path.append("../featureGen/TIs")
sys.path.append('../featureGen/originalData/data/futures/um/monthly/klines')

from Make_concat_sequenced import run_concat_sequenced
import LoadDfs
from Create_train_valid_test import TRAIN_VALID_TEST

from Model_classes import RF_Model
from Model_classes import LR_Model
#from Model_classes import NN_Model
#from Model_classes import NN_Model2
#from Model_classes import RNN_Model


############################# PIPELINE PROCEDURES ##############################
#Load original Dataframes
#Run_concat_sequenced 
#Run create_train_valid_test
#import models and run models on new data
#Run ensemble on  model pedictions
#Run simulation of ensemble predictions


def read(strategy_number, open_close = False):
    if open_close:
        add = "_open_close"
    else:
        add = ""
    dotusdt_df = pd.read_csv(f'../featureGen/PROCESSED_COINS/STRATEGY{strategy_number}/DF_5_Candles_Concat_DOTUSDT{add}.csv')
    ftmusdt_df = pd.read_csv(f'../featureGen/PROCESSED_COINS/STRATEGY{strategy_number}/DF_5_Candles_Concat_FTMUSDT{add}.csv')
    icpusdt_df = pd.read_csv(f'../featureGen/PROCESSED_COINS/STRATEGY{strategy_number}/DF_5_Candles_Concat_ICPUSDT{add}.csv')
    maticusdt_df = pd.read_csv(f'../featureGen/PROCESSED_COINS/STRATEGY{strategy_number}/DF_5_Candles_Concat_MATICUSDT{add}.csv')
    omgusdt_df = pd.read_csv(f'../featureGen/PROCESSED_COINS/STRATEGY{strategy_number}/DF_5_Candles_Concat_OMGUSDT{add}.csv')
    solusdt_df = pd.read_csv(f'../featureGen/PROCESSED_COINS/STRATEGY{strategy_number}/DF_5_Candles_Concat_SOLUSDT{add}.csv')
    coins_dfs = {'DOTUSDT':dotusdt_df, 'FTMUSDT':ftmusdt_df, 'ICPUSDT':icpusdt_df, 'MATICUSDT':maticusdt_df, 'OMGUSDT':omgusdt_df, 'SOLUSDT':solusdt_df}

    return coins_dfs

def save(strategy_number, train1, valid1, train2, valid2, test, open_close = False):
    if not os.path.isdir(f'../featureGen/PROCESSED_COINS/STRATEGY{strategy_number}/CONCAT_TRAIN_VALID_TEST'):
        os.makedirs(f'../featureGen/PROCESSED_COINS/STRATEGY{strategy_number}/CONCAT_TRAIN_VALID_TEST')
        
    if open_close:
        add = "_open_close"
    else:
        add = ""
    #Saving concatenated dataframes
    train1.to_csv(f'../featureGen/PROCESSED_COINS/STRATEGY{strategy_number}/CONCAT_TRAIN_VALID_TEST/DF_5_C_TRAIN_WEAK_LEARNERS{add}.csv',index=False)
    valid1.to_csv(f'../featureGen/PROCESSED_COINS/STRATEGY{strategy_number}/CONCAT_TRAIN_VALID_TEST/DF_5_C_VALID_WEAK_LEARNERS{add}.csv', index=False)
    train2.to_csv(f'../featureGen/PROCESSED_COINS/STRATEGY{strategy_number}/CONCAT_TRAIN_VALID_TEST/DF_5_C_TRAIN_ENSEMBLE{add}.csv',index=False)
    valid2.to_csv(f'../featureGen/PROCESSED_COINS/STRATEGY{strategy_number}/CONCAT_TRAIN_VALID_TEST/DF_5_C_VALID_ENSEMBLE{add}.csv', index=False)
    test.to_csv(f'../featureGen/PROCESSED_COINS/STRATEGY{strategy_number}/CONCAT_TRAIN_VALID_TEST/DF_5_C_TEST{add}.csv', index=False)


coins_dfs = read(2)
#Combine all coin dataframes, randomises rows, splits into train,valid and test for individual and ensemble model training 
whole_coin_df = TRAIN_VALID_TEST(coins_dfs)
train1, valid1, train2, valid2, test = whole_coin_df.run()
#Save the new combined dataframes
save(strategy,train1, valid1, train2, valid2, test)


def read_whole_coin_df(strategy_number, open_close = False):
    if open_close:
        add = "_open_close"
    else:
        add = ""
    train_concat = pd.read_csv(f'../featureGen/PROCESSED_COINS/STRATEGY{strategy_number}/CONCAT_TRAIN_VALID_TEST/DF_5_C_TRAIN_WEAK_LEARNERS{add}.csv')
    valid_concat = pd.read_csv(f'../featureGen/PROCESSED_COINS/STRATEGY{strategy_number}/CONCAT_TRAIN_VALID_TEST/DF_5_C_VALID_WEAK_LEARNERS{add}.csv')
    test_concat = pd.read_csv(f'../featureGen/PROCESSED_COINS/STRATEGY{strategy_number}/CONCAT_TRAIN_VALID_TEST/DF_5_C_TEST{add}.csv')
    return train_concat,valid_concat, test_concat



################## PIPELINE EXECUTION #####################


coins = LoadDfs.create_dataframes()

strategy = 1 
run_concat_sequenced(coins, strategy)

train_concat,valid_concat,test_concat = read_whole_coin_df(strategy)

RF = RF_Model.RF(train_concat, valid_concat, test_concat)
RF.run_model() 
#RF.save_model('trained_models_2/RF_model_1_EE')
#RF.load_model('trained_models_2/RF_model_1_EE')

print(test_concat["label"].tolist().count(0))
print(test_concat["label"].tolist().count(1))
print(test_concat["label"].tolist().count(2))
print(test_concat["label"].tolist().count(3))


preds = RF.model.predict_proba(RF.X_test)
RF.confusion_matrix(RF.X_test,RF.y_test)


########### Keeping track of the same preprocessed dataframes but with open, close, high, low columns without normalisation

run_concat_sequenced(coins, strategy, open_close = True)

coins_dfs_OC = read(strategy, open_close = True)
#Combine all coin dataframes, randomises rows, splits into train,valid and test for individual and ensemble model training 
whole_coin_df_OC = TRAIN_VALID_TEST(coins_dfs_OC)
train1_OC, valid1_OC, train2_OC, valid2_OC, test_OC = whole_coin_df_OC.run()
#Save the new combined dataframes
save(strategy,train1_OC, valid1_OC, train2_OC, valid2_OC, test_OC, open_close = True)

train_concat_OC,valid_concat_OC,test_concat_OC = read_whole_coin_df(strategy, open_close = True)


# Working out true positives and false positives based on minimum preset probabilities
confidences = [n/100 for n in range(30,55)]
for confidence in confidences:
    total_prediction_label_0 = 0
    TP, FP_label_0, FP_label_1, FP_label_2 = 0,0,0,0
    for i,pred_label in enumerate(preds):
        if pred_label[3] > confidence:
            total_prediction_label_0 +=1
            if RF.y_test.values[i] == 3:
                TP +=1
            elif RF.y_test.values[i] == 0:
                FP_label_0 +=1
            elif RF.y_test.values[i] == 1:
                FP_label_1 +=1
            elif RF.y_test.values[i] == 2:
                FP_label_2 +=1
            else:
                print("something is wrong")

    print(confidence)
    print(f"total amount predicted = {total_prediction_label_0}")
    print(f"true positives = {TP/total_prediction_label_0}")
    print(f"false positives label 0 = {FP_label_0/total_prediction_label_0}")
    print(f"false positives label 1 = {FP_label_1/total_prediction_label_0}")
    print(f"false positives label 2 = {FP_label_2/total_prediction_label_0}")


    
    
    
    
    
########### SIMULATION WARNING ########## 
### The simulation is dependent on the labels and strategy used, therefore this one is subject to change upon strategy ammendments #######
# Grid Search        
#gains = [i/1000 for i in range(1006, 1007)] # % gains upon entry prices
#losses = [i/1000 for i in range(994, 995)] # % loss upon entry prices
gains = [i/1000 for i in range(1004, 1010)] # % gains upon entry prices
losses = [i/1000 for i in range(985, 999)] # % loss upon entry prices

for gain in gains:
    for loss in losses:
        confidences = [n/100 for n in range(20,40,2)]
        for confidence in confidences:
            maker = 0.0002 # limit order
            taker = 0.0004 # market order
            #gain = 1.006   # gain of 0.6%
            #loss = 0.997 # loss of 0.3%
            fuckup_counter = 0
            true_pos_counter = 0
            num_trades = 0
            balance = [1000]
            assumptions = 0
            
            if gain == 1.008 and loss == 0.985 and confidence == 0.36:


                for i, pred in enumerate(preds):
                    # Check if the prediction is for the last row, if so ignore, because there is no subsequent row with data to be compared with
                    if i+1 < len(preds):
                        # Consider to get into trades where the label 3 has a predicted probability score over the param confidence
                        if pred[3] > confidence:
                            # Check if the actual label is 0, as this one would ultimately result in break even ASSUMING there is a balance of losses and winnings
                            if RF.y_test.values[i] != 0:
                                num_trades += 1
                                # Buying balance is the actual amount of money worth of a bought coin after fees are reduced
                                buying_balance = balance[-1]*(1-maker)

                                next_candle_high = test_concat_OC['high'][i+1] 
                                next_candle_low = test_concat_OC['low'][i+1] 
                                next_candle_open = test_concat_OC['open'][i+1]
                                next_candle_close = test_concat_OC['close'][i+1]

                                # If actual label is 3
                                if RF.y_test.values[i] == 3:
                                    true_pos_counter += 1
                                    if (next_candle_high/next_candle_open) >= gain or (next_candle_low/next_candle_open) <= (2-gain):
                                        updated_balance = (((gain*next_candle_open)/(next_candle_open*1.003))*buying_balance)*(1-maker)
                                        print(f'Longing in Label 3 (getting profits): From {round(balance[-1],2)} to {round(updated_balance,2)}')
                                        assumptions += 1
                                        #print(f'buying balance: {round(buying_balance,2)} -- open: {next_candle_open} -- close: {next_candle_close} -- high: {next_candle_high} -- low: {next_candle_low}')
                                    else:
                                        updated_balance = (buying_balance*loss)*(1-maker)
                                        fuckup_counter += 1 
                                        print(f'Longing/Shorting in Label 3 (getting losses): From {round(balance[-1],2)} to {round(updated_balance,2)}')
                                    balance.append(updated_balance)

                                # If actual label is 2        
                                if RF.y_test.values[i] == 2: 
                                    if (next_candle_low/next_candle_open) <= (2-gain):
                                        updated_balance =  (((1-(2-gain)/0.997)+1)*buying_balance)*(1-maker)
                                        print(f'Shorting in Label 2 (getting profits): From {round(balance[-1],2)} to {round(updated_balance,2)}')
                                        print(f'buying balance: {round(buying_balance,2)} -- open: {next_candle_open} -- close: {next_candle_close} -- high: {next_candle_high} -- low: {next_candle_low}')

                                    else:
                                        updated_balance = (buying_balance*loss)*(1-maker)
                                        print(f'Shorted but it went long in Label 2 (getting losses): From {round(balance[-1],2)} to {round(updated_balance,2)}')
                                        #print(f'buying balance: {round(buying_balance,2)} -- open: {next_candle_open} -- close: {next_candle_close} -- high: {next_candle_high} -- low: {next_candle_low}')

                                    balance.append(updated_balance)


                                # If actual label is 1
                                if RF.y_test.values[i] == 1:
                                    if (next_candle_high/next_candle_open) >= gain:
                                        updated_balance =  ((gain/1.003)*buying_balance)*(1-maker)
                                        print(f'Longing in Label 1 (getting profits): From {round(balance[-1],2)} to {round(updated_balance,2)}')
                                        #print(f'buying balance: {round(buying_balance,2)} -- open: {next_candle_open} -- close: {next_candle_close} -- high: {next_candle_high} -- low: {next_candle_low}')

                                    else:
                                        updated_balance = (buying_balance*loss)*(1-maker)
                                        print(f'Longed but it went short in Label 1 (getting losses): From {round(balance[-1],2)} to {round(updated_balance,2)}')
                                        #print(f'buying balance: {round(buying_balance,2)} -- open: {next_candle_open} -- close: {next_candle_close} -- high: {next_candle_high} -- low: {next_candle_low}')

                                    balance.append(updated_balance)

                #print([round(bal,2) for bal in balance])
                print(f'Assumptions: {assumptions}')
                print(f'Gain: {gain} // Loss: {loss}')
                print(f'Confidence: {confidence}')    
                print(f'Fuckups: {fuckup_counter}')
                print(f'Trades: {num_trades} -- True positives: {true_pos_counter} ({round((true_pos_counter/num_trades)*100,2)}%)')
                print(f'Initial balance: {balance[0]} -- Final balance: {balance[-1]} \n')
                print(dd)
                
                




LR = LR_Model.LR(train_concat, valid_concat, test_concat)
LR.run_model()


nn1 = NN_Model.NN(train_concat, valid_concat, test_concat)
nn1_model = nn1.run_model()


nn2 = NN_Model2.NN2(train_concat, valid_concat, test_concat)
nn2_model = nn2.run_model()


#Run RNN


#Run ensemble


#Run simulation on ensemble




