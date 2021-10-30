#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
pd.options.display.max_columns = None
import requests
np.set_printoptions(suppress=True)
import tensorflow
from numpy import loadtxt
from tensorflow import keras
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from keras.losses import CategoricalCrossentropy


# In[13]:


class NN:
    
    def __init__(self,train, valid, test):
        self.train_data = train
        self.valid_data = valid
        self.test_data = test
        self.run()
        
    def run(self):
        #Method that will run on when class is called
        self.X_train, self.y_train = self.prepare_data(self.train_data)
        self.X_valid, self.y_valid = self.prepare_data(self.valid_data)
        self.X_test, self.y_test = self.prepare_data(self.test_data)
        
    def prepare_data(self,data):
        #Method that prepares data into X, Y
        limit = data.values[:,:-1].shape[1]
        X = data.values[:,0:limit]
        y = data.values[:,limit]
        y = self.encode(y)
        return X,y
        
    
    def encode(self,y):
        #Method that converts integers to dummy variables (i.e. one hot encoded)
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_Y = encoder.transform(y)
        y = np_utils.to_categorical(encoded_Y)
        return y

    def run_model(self):
        main_model = Sequential()
        main_model.add(Dense(128, input_dim=self.X_train.shape[-1], activation='relu'))
        main_model.add(Dropout(0.2))
        main_model.add(BatchNormalization())

        main_model.add(Dense(32, activation='relu'))
        main_model.add(Dropout(0.1))

        main_model.add(Dense(32,activation='relu'))
        main_model.add(Dropout(0.2))
        main_model.add(BatchNormalization())

        main_model.add(Dense(32, activation='relu'))
        main_model.add(Dropout(0.1))

        main_model.add(Dense(32, activation='relu'))
        main_model.add(Dropout(0.2))
        main_model.add(BatchNormalization())

        main_model.add(Dense(32, activation='relu'))
        main_model.add(Dropout(0.1))

        main_model.add(Dense(4, activation='softmax'))

        optimizer = keras.optimizers.Adam(learning_rate=0.001)

        main_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy',"MSE"])
        main_model.fit(self.X_train, self.y_train, epochs=15, validation_data =(self.X_valid,self.y_valid), validation_split =0.2)
    
        preds = main_model.predict(self.X_test)
        self.get_accuracy(preds)
        return main_model
    
    def get_accuracy(self, predictions):
        
        predictions_full = []
        for each in predictions:
            ind = np.argmax(each)
            predictions_full.append(ind)
        real_values = []
        for each in self.y_test:
            ind = np.argmax(each)
            real_values.append(ind)

        print("Length match up test:", len(predictions)==len(self.y_test), len(predictions_full)== len(real_values))
        accuracy = 0
        for i in range(len(predictions_full)):
            if predictions_full[i] == real_values[i]:
                accuracy += 1

        print("Test Accuracy: ",accuracy/len(predictions_full))
        
    


# In[ ]:




