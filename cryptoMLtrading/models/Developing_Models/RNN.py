#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


test_df = pd.read_csv('../featureGen/PROCESSED_COINS/Sequenced/DF_sequence_DOTUSDT.csv')
validation_main = test_df[100000:]
test_df = test_df[:50000]


# In[3]:


#%tensorflow_version 1.x


# In[4]:


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


# In[5]:


X_train, Y_train = preprocess(test_df)
X_valid, Y_valid = preprocess(validation_main)


# In[6]:


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


# In[7]:


Y_train = encode(Y_train)
Y_valid = encode(Y_valid)

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
X_valid = np.asarray(X_valid)
Y_valid = np.asarray(Y_valid)


# ## Grid Search on RNN

# In[8]:


layers = 4
nodes = [32,64,128,256]
dropout = [0,0.1,0.2]
EPOCHS = [5,10,20]
BATCH_SIZE = [32,64]
time = 0
for node in nodes:
    for i in range(layers):
        for drop in dropout:
            for epoch in EPOCHS:
                print(f"Run Number  {time} out of 600 approx")

                model = Sequential()
                model.add(CuDNNLSTM(node, input_shape=(X_train.shape[1:]), return_sequences=True))
                model.add(Dropout(drop))
                model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.
                
                if i > 0:
                    model.add(CuDNNLSTM(node, input_shape=(X_train.shape[1:]), return_sequences=True))
                    model.add(Dropout(drop))
                    model.add(BatchNormalization())
                
                if i > 1:

                    model.add(CuDNNLSTM(node, input_shape=(X_train.shape[1:]), return_sequences=True))
                    model.add(Dropout(drop))
                    model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.
                
                if i > 2:
                    
                    model.add(CuDNNLSTM(node, input_shape=(X_train.shape[1:]), return_sequences=True))
                    model.add(Dropout(drop))
                    model.add(BatchNormalization())
                    
                model.add(CuDNNLSTM(node, input_shape=(X_train.shape[1:])))
                model.add(Dropout(drop))
                model.add(BatchNormalization())

                model.add(Dense(32, activation='relu'))
                model.add(Dropout(drop))

                model.add(Dense(4,activation='softmax'))

                #opt = tf.keras.optimizers.Adam(lr = 0.001, decay = 1e-5)

                model.compile(loss = "categorical_crossentropy",
                             optimizer = 'adam',
                             metrics = ["accuracy"])


                #tensorboard = TensorBoard(log_dir = )
                print(f"Node: {node}, layer: {i+3}, drop: {drop}, Epochs: {epochs}")

                
                model.fit(X_train, Y_train, batch_size = 64, epochs = epoch, validation_data = (X_valid, Y_valid),verbose = 0)

                #model.save("/content/drive/My Drive/models/RNN_model.h5")
                preds = model.predict(X_valid)

                len(preds) == len(Y_valid)
                predictions_full = []
                for each in preds:
                    ind = np.argmax(each)
                    predictions_full.append(ind)
                real_values = []
                for each in Y_valid:
                    ind = np.argmax(each)
                    real_values.append(ind)

                #print(len(real_values)==  len(predictions_full))
                accuracy = 0
                for i in range(len(predictions_full)):
                    if predictions_full[i] == real_values[i]:
                        accuracy += 1
                time+=1
                #print(f"Node: {node}, layer: {layers}, drop: {dropout}, Epochs: {epochs}")
                print("accuracy", accuracy/len(predictions_full), "\n")


# In[10]:



model = Sequential()
model.add(CuDNNLSTM(32, input_shape=(X_train.shape[1:]), return_sequences=True))
model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

model.add(CuDNNLSTM(32, input_shape=(X_train.shape[1:]), return_sequences=True))
model.add(BatchNormalization())

model.add(CuDNNLSTM(32, input_shape=(X_train.shape[1:]), return_sequences=True))
model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.


model.add(CuDNNLSTM(32, input_shape=(X_train.shape[1:])))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dense(4,activation='softmax'))

#opt = tf.keras.optimizers.Adam(lr = 0.001, decay = 1e-5)

model.compile(loss = "categorical_crossentropy",
             optimizer = 'adam',
             metrics = ["accuracy"])


#tensorboard = TensorBoard(log_dir = )
print(f"Node: {32}, layer: {2}, drop: {0.2}, Epochs: {20}")


model.fit(X_train, Y_train, batch_size = 64, epochs = 10, validation_data = (X_valid, Y_valid),verbose = 1)

preds = model.predict(X_valid)

len(preds) == len(Y_valid)
predictions_full = []
for each in preds:
    ind = np.argmax(each)
    predictions_full.append(ind)
real_values = []
for each in Y_valid:
    ind = np.argmax(each)
    real_values.append(ind)

#print(len(real_values)==  len(predictions_full))
accuracy = 0
for i in range(len(predictions_full)):
    if predictions_full[i] == real_values[i]:
        accuracy += 1
time+=1
#print(f"Node: {node}, layer: {layers}, drop: {dropout}, Epochs: {epochs}")
print("accuracy", accuracy/len(predictions_full), "\n")


# In[13]:


model.save("../models/trained_models_2/RNN_model_1_EE")


# In[11]:


test_df.shape


# In[12]:


preds


# In[ ]:


def true_label(y_test):
    #Turn the real labels into their index positions
    real_values = []
    #before test_df
    for each in y_test:
        ind = np.argmax(each)
        real_values.append(ind)
    return real_values

real_values = true_label(Y_valid)


# In[ ]:


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


# In[ ]:


predictions = model.predict(X_valid)
maker = 0.00018
taker = 0.00018
starting = 1000
leverage = 1
confidence = []

for i in range(20,80,2):
    confidence.append(i/100)
for confidence in confidence:   
    simulation(predictions,real_values,maker,taker, starting, leverage, confidence)


# In[ ]:




