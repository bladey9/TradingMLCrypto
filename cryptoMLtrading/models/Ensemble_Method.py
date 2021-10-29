#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from collections import deque
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils import np_utils

from tensorflow import keras
from tensorflow.python.keras.layers import Input, Dense,BatchNormalization, Dropout
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


# In[2]:


# Loading concatenated dataframes
train_concat = pd.read_csv('../featureGen/PROCESSED_COINS/CONCAT_ALL_COINS/DF_5_Candles_Concat_ALL_COINS_TRAIN_ENSEMBLE.csv')
valid_concat = pd.read_csv('../featureGen/PROCESSED_COINS/CONCAT_ALL_COINS/DF_5_Candles_Concat_ALL_COINS_VALID_ENSEMBLE.csv')
test_concat = pd.read_csv('../featureGen/PROCESSED_COINS/CONCAT_ALL_COINS/DF_5_Candles_Concat_ALL_COINS_TEST.csv')

# Loading sequenced dataframes
train_sequenced = pd.read_csv('../featureGen/PROCESSED_COINS/Sequenced_ALL_COINS/DF_5_Candles_Sequenced_ALL_COINS_TRAIN_ENSEMBLE.csv')
valid_sequenced = pd.read_csv('../featureGen/PROCESSED_COINS/Sequenced_ALL_COINS/DF_5_Candles_Sequenced_ALL_COINS_VALID_ENSEMBLE.csv')
test_sequenced = pd.read_csv('../featureGen/PROCESSED_COINS/Sequenced_ALL_COINS/DF_5_Candles_Sequenced_ALL_COINS_TEST.csv')


# In[3]:


def split(dataframe):
    # Split X and y data into 2 dataframes
    X_train = dataframe.loc[:, dataframe.columns != 'label']
    #X.head()
    y_train = dataframe.loc[:, dataframe.columns == 'label']
    #y.head()
    return X_train, y_train


# In[4]:


def encode(Y):
    #Method to one-hot encode vectors
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    
    return dummy_y

def preprocess(test_df):
    sequential_data = []
    prev_days = deque(maxlen=5)
    for i in test_df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == 5:
            sequential_data.append([np.array(prev_days),i[-1]])
    X = []
    y = []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y


# In[6]:


#Concat data
X_train_concat, y_train_concat = split(train_concat[4:])
X_valid_concat, y_valid_concat = split(valid_concat[4:])
X_test_concat, y_test_concat = split(test_concat[4:])

#RNN sequenced data
X_train_EE_RNN,y_train_EE_RNN = preprocess(train_sequenced)
X_valid_EE_RNN, y_valid_EE_RNN = preprocess(valid_sequenced)
X_test_EE_RNN,y_test_EE_RNN = preprocess(test_sequenced)

#NN label encoding
y_train_EE_concat_NN = encode(y_train_concat)
y_valid_EE_concat_NN = encode(y_valid_concat)
y_test_EE_concat_NN = encode(y_test_concat)

y_train_EE_RNN = encode(y_train_EE_RNN)
y_valid_EE_RNN = encode(y_valid_EE_RNN)
y_test_EE_RNN = encode(y_test_EE_RNN)


# In[10]:


print(len(X_test_concat) == len(X_test_EE_RNN))
print(len(y_train_concat) == len(y_train_EE_RNN))
print(len(y_test_concat) == len(y_test_EE_RNN))
print(len(X_valid_concat) == len(X_valid_EE_RNN))
print(len(y_valid_concat) == len(y_valid_EE_RNN))


# In[11]:


RF = pickle.load(open('trained_models_2/RF_model_1_EE.sav', 'rb'))
LR = pickle.load(open('trained_models_2/LR_model_1_EE.sav', 'rb'))
NN1 = keras.models.load_model('trained_models_2/NN_model_1_All_coins_EE')
NN2 = keras.models.load_model('trained_models_2/NN_model_2_All_coins_EE')
RNN = keras.models.load_model('trained_models_2/RNN_model_1_ALL_COINS_EE')


# In[12]:


RF_preds_for_ensemble = RF.predict_proba(X_train_concat.values)
print('RF DONE')
LR_preds_for_ensemble = LR.predict_proba(X_train_concat.values)
print('LR DONE')
NN1_preds_for_ensemble = NN1.predict(X_train_concat.values)
print('NN1 DONE')
NN2_preds_for_ensemble = NN2.predict(X_train_concat.values)
print('NN2 DONE')
RNN_preds_for_ensemble = RNN.predict(X_train_EE_RNN)
print('RNN DONE')


# In[16]:


RF_preds_for_ensemble_valid = RF.predict_proba(X_valid_concat.values)
LR_preds_for_ensemble_valid  = LR.predict_proba(X_valid_concat.values)
NN1_preds_for_ensemble_valid  = NN1.predict(X_valid_concat.values)
NN2_preds_for_ensemble_valid  = NN2.predict(X_valid_concat.values)
RNN_preds_for_ensemble_valid  = RNN.predict(X_valid_EE_RNN)


# In[13]:


RF_preds_for_ensemble_test = RF.predict_proba(X_test_concat.values)
LR_preds_for_ensemble_test = LR.predict_proba(X_test_concat.values)
NN1_preds_for_ensemble_test = NN1.predict(X_test_concat.values)
NN2_preds_for_ensemble_test = NN2.predict(X_test_concat.values)
RNN_preds_for_ensemble_test = RNN.predict(X_test_EE_RNN)


# In[14]:


models_E = [RF_preds_for_ensemble,LR_preds_for_ensemble,NN1_preds_for_ensemble,NN2_preds_for_ensemble,RNN_preds_for_ensemble]
models_test = [RF_preds_for_ensemble_test,LR_preds_for_ensemble_test,NN1_preds_for_ensemble_test,NN2_preds_for_ensemble_test,RNN_preds_for_ensemble_test]
for model in models_test:
    predictions_full = []
    for each in model:
        ind = np.argmax(each)
        predictions_full.append(ind)
        
    real_values = []
    for each in y_test_EE_concat_NN:
        ind = np.argmax(each)
        real_values.append(ind)
    
    accuracy = 0
    print(len(predictions_full) == len(real_values))
    for i in range(len(predictions_full)):
        if predictions_full[i] == real_values[i]:
            accuracy+=1
    print(f"Accuracy of model = {accuracy/len(predictions_full)}")
    


# In[15]:


RF_preds_for_ensemble.shape == LR_preds_for_ensemble.shape == NN1_preds_for_ensemble.shape == NN2_preds_for_ensemble.shape == RNN_preds_for_ensemble.shape


# In[17]:


concat_preds_EE_train = np.concatenate((RF_preds_for_ensemble,LR_preds_for_ensemble,NN1_preds_for_ensemble,NN2_preds_for_ensemble,RNN_preds_for_ensemble), axis=1)
concat_preds_EE_valid = np.concatenate((RF_preds_for_ensemble_valid,LR_preds_for_ensemble_valid,NN1_preds_for_ensemble_valid,NN2_preds_for_ensemble_valid,RNN_preds_for_ensemble_valid), axis=1)
concat_preds_EE_test = np.concatenate((RF_preds_for_ensemble_test,LR_preds_for_ensemble_test,NN1_preds_for_ensemble_test,NN2_preds_for_ensemble_test,RNN_preds_for_ensemble_test), axis=1)


# In[18]:



main_model = Sequential()
main_model.add(Dense(64, input_dim=concat_preds_EE_train.shape[1], activation='relu'))
main_model.add(Dropout(0.2))

main_model.add(Dense(32, input_dim=64, activation='relu'))
main_model.add(Dropout(0.2))

main_model.add(Dense(16, input_dim=16, activation='relu'))
main_model.add(Dropout(0.2))

main_model.add(Dense(4, activation='softmax'))

#optimizer = keras.optimizers.Adam(lr=0.001)

main_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',"MSE"])
main_model.fit(concat_preds_EE_train, y_train_EE_concat_NN, epochs=5, validation_data =(concat_preds_EE_valid,y_valid_EE_concat_NN), validation_split =0.1)

predictions = main_model.predict(concat_preds_EE_test)

print(len(predictions) == len(y_test_EE_concat_NN))
predictions_full = []
for each in predictions:
    ind = np.argmax(each)
    
    predictions_full.append(ind)
real_values = []
for each in y_test_EE_concat_NN:
    ind = np.argmax(each)
    real_values.append(ind)

print(len(real_values)==  len(predictions_full))
accuracy = 0
for i in range(len(predictions_full)):
    if predictions_full[i] == real_values[i]:
        accuracy += 1


print(accuracy/len(predictions_full))


# In[19]:


main_model.save("trained_models_2/ensemble_1")


# In[ ]:




