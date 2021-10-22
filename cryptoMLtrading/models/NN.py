#!/usr/bin/env python
# coding: utf-8

# In[140]:


import numpy as np
import pandas as pd
#pd.options.display.max_rows = None
pd.options.display.max_columns = None
import requests
np.set_printoptions(suppress=True)
from numpy import loadtxt
from tensorflow import keras
from tensorflow.python.keras.layers import Input, Dense,BatchNormalization
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


# In[3]:


test_df = pd.read_csv("df_for_model_3.0")


# In[69]:


def prepare_data(dataset):
    dataset.to_csv(r'NNdata.csv',header = False, index = False)
    dataset = loadtxt(r'NNdata.csv', delimiter=',')
    X = dataset[:,0:120].astype(float)
    Y = dataset[:,120]
    
    return X,Y


# In[70]:


def encode(Y):
    #Method to one-hot encode vectors
    #@params Y Labels
    #@return Y_labels encoded
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    
    return dummy_y


# In[71]:



def split(X,Y, div = 0.8):
    division = int(len(X)*div)
    X_train = X[:division,:]
    X_test = X[division:,:]
    Y_train = Y[:division]
    Y_test = Y[division:]
    return X_train,X_test,Y_train, Y_test 


# In[79]:


X,Y = prepare_data(test_df)
Y = encode(Y)
X_train,X_test,Y_train, Y_test = split(X,Y)


# In[80]:


#Testing script
(len(X_train) + len(X_test)) == len(X)


# In[ ]:


main_model 


# In[143]:


main_model = Sequential()
main_model.add(Dense(240, input_dim=120, activation='relu'))
main_model.add(BatchNormalization())
main_model.add(Dense(240, input_dim=120, activation='relu'))
main_model.add(BatchNormalization())
main_model.add(Dense(4, activation='softmax'))

main_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',"MSE"])
early_stopping_monitor = EarlyStopping(patience=10)
main_model.fit(X_train, Y_train, epochs=100, validation_split=0.2)

predictions = main_model.predict(X_test)


# In[151]:


#main_model.save("NN_model_1.0")


# In[145]:


def true_label(y_test):
    #Turn the real labels into their index positions
    real_values = []
    #before test_df
    for each in y_test:
        ind = np.argmax(each)
        real_values.append(ind)
    return real_values


# In[146]:


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
    print("Ratio of Label 1 to Label 4: {}".format(amount_1/amount_4))
    plt.figure(figsize=(12,6))
    plt.plot(balances)


# In[148]:


import matplotlib.pyplot as plt
real_values = true_label(Y_test)
maker = 0.00018
taker = 0.00018
starting = 1000
leverage = 1
confidence = 0.6
confidence =[0.4,0.5,0.6,0.7,0.8,0.9,1.0]
for confidence in confidence:   
    simulation(predictions,real_values,maker,taker, starting, leverage, confidence)


# In[84]:


def testing(preds, truth, taker, maker, gain, normal, loss):
    if len(preds)== len(truth):
        print("Test 1 Success")
    else:
        print("Test 1 Failed, length of arrays MISSMATCH")
        return False
    
    if taker / maker == 2:
        print("Test 2 Success")
    else:
        print("Test 2 Failed - taker and maker are not equal (relative)")
        return False
    
    if (round((1+gain) - normal,3)) == 0.005:
        print("Test 3 Success")
    else:
        
        print("Test 3 Failed")
        return False
    
    if (round(normal - (1-loss),3)) == 0.003:
        print("Test 4 Success")
    else:
        print("Test 4 Failed")
        return False
    
    if (taker * 100) > 0.01 and (taker * 100) < 0.1:
        print("Test 5 Success")
    else:
        print("Test 5 Failed - Taker is wrong")
        return False
    
    if (maker * 100) > 0.01 and (maker * 100) < 0.1:
        print("Test 6 Success")
    else:
        print("Test 6 Failed - Maker is wrong")
        return False
    
    return True

