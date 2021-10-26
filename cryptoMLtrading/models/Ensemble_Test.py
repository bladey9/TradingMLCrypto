#!/usr/bin/env python
# coding: utf-8

# In[25]:


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
pd.options.display.max_columns = None


# In[42]:


concat_data = pd.read_csv('../featureGen/PROCESSED_COINS/Concat/DF_5_Candles_Concat_SOLUSDT.csv.zip')
sequenced_data = pd.read_csv('../featureGen/PROCESSED_COINS/Sequenced/DF_sequence_SOLUSDT.csv')
sequenced_data = sequenced_data[1:]


# In[ ]:





# In[43]:


# Split X and y data into 2 dataframes
X_concat = concat_data.loc[:, concat_data.columns != 'label']
y_concat = concat_data.loc[:, concat_data.columns == 'label']


# In[64]:


rv = []
for each in y_concat:
    ind = np.argmax(each)
    rv.append(ind)
    
print(rv.count(0)/len(y_concat))

print(rv.count(1)/len(y_concat))

print(rv.count(2)/len(y_concat))

print(rv.count(3)/len(y_concat))


# In[35]:



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


# In[45]:


#RNN sequenced data
X_seq,y_seq = preprocess(sequenced_data)

X_concat = X_concat.head(X_concat.shape[0])
y_concat = y_concat.head(y_concat.shape[0])

#NN label encoding
y_concat = encode(y_concat)
y_seq = encode(y_seq)


# In[46]:


print(len(X_concat) == len(X_seq))
print(len(y_concat) == len(y_seq))


# In[53]:


RF = pickle.load(open('trained_models_2/RF_model_1_EE.sav', 'rb'))
#print(f'Random Forest Accuracy Score: {RF.score(X_test, y_test)}')
LR = pickle.load(open('trained_models_2/LR_model_1_EE.sav', 'rb'))
#print(f'Logistic Regression Accuracy Score: {LR.score(X_test, y_test)}')
NN1 = keras.models.load_model('trained_models_2/NN_model_1_EE')
NN2 = keras.models.load_model('trained_models_2/NN_model_2_EE')
RNN = keras.models.load_model('trained_models_2/RNN_model_2_EE')
#SVM = pickle.load(open('trained_models/SVM_model_1.0.sav', 'rb'))
#print(f'SVM Accuracy Score: {SVM.score(X_test, y_test)}')


# In[55]:


ensemble = keras.models.load_model("trained_models_2/ensemble_1")


# In[56]:


RF_preds_for_ensemble = RF.predict_proba(X_concat.values)
LR_preds_for_ensemble = LR.predict_proba(X_concat.values)
NN1_preds_for_ensemble = NN1.predict(X_concat.values)
NN2_preds_for_ensemble = NN2.predict(X_concat.values)
RNN_preds_for_ensemble = RNN.predict(X_seq)

concat_preds_EE = np.concatenate((RF_preds_for_ensemble,LR_preds_for_ensemble,NN1_preds_for_ensemble,NN2_preds_for_ensemble,RNN_preds_for_ensemble), axis=1)
ensemble_preds = ensemble.predict(concat_preds_EE)


# In[58]:


models = {"RF":RF_preds_for_ensemble,"LR":LR_preds_for_ensemble,"NN1":NN1_preds_for_ensemble,"NN2": NN2_preds_for_ensemble,"RNN":RNN_preds_for_ensemble, "Ensemble":ensemble_preds}

for key, values in models.items():
    predictions_full = []
    for each in values:
        ind = np.argmax(each)
        predictions_full.append(ind)
        
    real_values = []
    for each in y_concat:
        ind = np.argmax(each)
        real_values.append(ind)
    
    accuracy = 0
    print(len(predictions_full) == len(real_values))
    for i in range(len(predictions_full)):
        if predictions_full[i] == real_values[i]:
            accuracy+=1
    print(f" Model {key}, Accuracy of model = {accuracy/len(predictions_full)}")
    
    
    


# In[127]:


RF_preds_for_ensemble.shape == LR_preds_for_ensemble.shape == NN1_preds_for_ensemble.shape == NN2_preds_for_ensemble.shape == RNN_preds_for_ensemble.shape


# In[132]:


concat_preds_EE_train = np.concatenate((RF_preds_for_ensemble,LR_preds_for_ensemble,NN1_preds_for_ensemble,NN2_preds_for_ensemble,RNN_preds_for_ensemble), axis=1)
concat_preds_EE_valid = np.concatenate((RF_preds_for_ensemble_valid,LR_preds_for_ensemble_valid,NN1_preds_for_ensemble_valid,NN2_preds_for_ensemble_valid,RNN_preds_for_ensemble_valid), axis=1)
concat_preds_EE_test = np.concatenate((RF_preds_for_ensemble_test,LR_preds_for_ensemble_test,NN1_preds_for_ensemble_test,NN2_preds_for_ensemble_test,RNN_preds_for_ensemble_test), axis=1)


# In[139]:



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
main_model.fit(concat_preds_EE_train, y_train_EE_concat_NN, epochs=20, validation_data =(concat_preds_EE_valid,y_valid_EE_concat_NN), validation_split =0.1)

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


# In[141]:


main_model.save("trained_models/ensemble_1")


# In[ ]:




