# -*- coding: utf-8 -*-
"""RNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1A239g7pny19jqeK0wpGDr4Kh7CTO2OGd
"""

import pandas as pd
import numpy as np
#pd.options.display.max_rows = None
pd.options.display.max_columns = None
import requests
np.set_printoptions(suppress=True)
from numpy import loadtxt
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Input, Dense,BatchNormalization, Dropout,LSTM, CuDNNLSTM
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint
import random
from collections import deque
import matplotlib.pyplot as plt

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x

#from google.colab import drive
#drive.mount('/content/drive')

#test_df = pd.read_csv("/content/drive/My Drive/df_for_model_2.0")
#validation_main = test_df[100000:]
#test_df = test_df[:100000]

def preprocess(test_df):
    sequential_data = []
    prev_days = deque(maxlen=5)
    for i in test_df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == 5:
            sequential_data.append([np.array(prev_days),i[-1]])
    
    random.shuffle(sequential_data)
    X = []
    y = []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y

#X_train, Y_train = preprocess(test_df)
#X_valid, Y_valid = preprocess(validation_main)

from sklearn.preprocessing import LabelEncoder
def encode(Y):
    #Method to one-hot encode vectors
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    
    return dummy_y

#Y_train = encode(Y_train)
#Y_valid = encode(Y_valid)

#X_train = np.asarray(X_train)
#Y_train = np.asarray(Y_train)
#X_valid = np.asarray(X_valid)
#Y_valid = np.asarray(Y_valid)

#EPOCHS = 10
#BATCH_SIZE = 65

#model = Sequential()
#model.add(CuDNNLSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

#model.add(CuDNNLSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
#model.add(Dropout(0.1))
#model.add(BatchNormalization())

#model.add(CuDNNLSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

#model.add(CuDNNLSTM(128, input_shape=(X_train.shape[1:])))
#model.add(Dropout(0.1))
#model.add(BatchNormalization())

#model.add(Dense(32, activation='relu'))
#model.add(Dropout(0.2))

#model.add(Dense(4,activation='softmax'))


#opt = tf.keras.optimizers.Adam(lr = 0.001, decay = 1e-5)

#model.compile(loss = "categorical_crossentropy",
#             optimizer = 'adam',
#             metrics = ["accuracy"])


#tensorboard = TensorBoard(log_dir = )

#model.fit(X_train, Y_train, batch_size = 64, epochs = 20, validation_data = (X_valid, Y_valid))

#model.save("/content/drive/My Drive/models/RNN_model.h5")

def true_label(y_test):
    #Turn the real labels into their index positions
    real_values = []
    #before test_df
    for each in y_test:
        ind = np.argmax(each)
        real_values.append(ind)
    return real_values

#real_values = true_label(Y_valid)

def simulation(preds, truth, maker, taker, starting, leverage, confidence = 0.7):

    loss = 0.003
    normal = 1.0
    gain = 0.005

    balance = starting

    preds = preds.tolist()

    #TESTING
    #if testing(preds, truth, taker, maker, gain, normal, loss) is False:
        #print("Testing Unsuccessful")
    
    
    gain = 1+ (gain * leverage)
    loss = (1- (loss* leverage))
    maker = maker * leverage
    taker = taker * leverage
    
    
    #See the truth labels of the predicted 0 labels with confidence higher than x
    amount_1 = 0
    amount_4 = 0
    truth_in_pred = []
    for i in range(len(preds)):

        if preds[i][0] >= confidence:
            
            alist = [0,truth[i]]
            truth_in_pred.append(alist)
        elif preds[i][3] >= confidence:
            alist = [3,truth[i]]
            truth_in_pred.append(alist)
    
    #Money Simulation
    balances = []   
    for i in range(len(truth_in_pred)):
        if truth_in_pred[i][0] == 0:
            amount_1 +=1
            balance = balance * (1-taker)
            if truth_in_pred[i][1] == 0:
                balance = (balance * gain)*(1-maker)
            elif truth_in_pred[i][1] == 1:
                balance = (balance * normal)* (1-maker)
            elif truth_in_pred[i][1] == 2:
                balance = (balance * normal)* (1-maker)
            elif truth_in_pred[i][1] == 3:
                balance = (balance * loss)* (1-maker)
            else:
                "MASSIVE ERROR"
                break
            balances.append(balance)
        elif truth_in_pred[i][0] == 3:
            amount_4 +=1
            balance = balance * (1-taker)
            if truth_in_pred[i][1] == 0:
                balance = (balance * loss)*(1-maker)
            elif truth_in_pred[i][1] == 1:
                balance = (balance * normal)* (1-maker)
            elif truth_in_pred[i][1] == 2:
                balance = (balance * normal)* (1-maker)
            elif truth_in_pred[i][1] == 3:
                balance = (balance * gain)* (1-maker)
            else:
                "MASSIVE ERROR"
                break
            balances.append(balance)
            

    print(f"Starting Balance= {starting}")
    print("--")
    print(f"Ending Balance = {balance}")
    print("Amount of Label 1 with a confidence score > {} = {} ".format(confidence, len(truth_in_pred)))
    print("Amount of True Label 1 to True Label 4= {}:{}".format(amount_1, amount_4))
    print("Ratio of Label 1 to Label 4: {}".format(amount_1,amount_4))
    plt.figure(figsize=(12,6))
    plt.plot(balances)

#predictions = model.predict(X_valid)
#maker = 0.00018
#taker = 0.00018
#starting = 1000
#leverage = 1
#confidence = []

#for i in range(20,80,2):
#    confidence.append(i/100)
#for confidence in confidence:   
#    simulation(predictions,real_values,maker,taker, starting, leverage, confidence)