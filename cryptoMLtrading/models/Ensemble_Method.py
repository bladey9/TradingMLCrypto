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
from RNN import preprocess, encode


# In[2]:


data = pd.read_csv('../featureGen/df_for_model_3.0.zip')
data.head()


# In[3]:


# This dataframe is used for the RNN since the model does not require data from X previous rows stacked
# together in a single column
data2 = pd.read_csv('../featureGen/df_for_model_2.0') 
data2.head()


# In[4]:


# Split X and y data into 2 dataframes
X = data.loc[:, data.columns != 'label']
#X.head()
y = data.loc[:, data.columns == 'label']
#y.head()


# In[5]:


X_train,X_test,y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)
print('X_train:', len(X_train))
print('y_train:', len(y_train), '\n')
print('X_test:', len(X_test))
print('y_test:', len(y_test))


# In[6]:


train_data_RNN = data2[:100000]
test_data_RNN = data2[100000:]

RNN_X_train, RNN_y_train = preprocess(train_data_RNN)
RNN_X_test, RNN_y_test = preprocess(test_data_RNN)

RNN_y_train = encode(RNN_y_train)
RNN_y_test = encode(RNN_y_test)


# In[7]:


RF = pickle.load(open('trained_models/RF_model_1.0.sav', 'rb'))
print(f'Random Forest Accuracy Score: {RF.score(X_test, y_test)}')
LR = pickle.load(open('trained_models/LR_model_1.0.sav', 'rb'))
print(f'Logistic Regression Accuracy Score: {LR.score(X_test, y_test)}')
NN1 = keras.models.load_model('trained_models/NN_model_1.0')
NN2 = keras.models.load_model('trained_models/NN_model_2.0')
RNN = keras.models.load_model('trained_models/RNN_model')
SVM = pickle.load(open('trained_models/SVM_model_1.0.sav', 'rb'))
print(f'SVM Accuracy Score: {SVM.score(X_test, y_test)}')


# In[ ]:


#RNN = keras.models.load_model('trained_models/RNN_model')
#RNN.predict(RNN_X_test)


# In[ ]:


#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())


# In[ ]:




