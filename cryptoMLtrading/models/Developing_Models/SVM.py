#!/usr/bin/env python
# coding: utf-8

# In[29]:


from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle


# In[2]:


data = pd.read_csv('../featureGen/df_for_model_3.0.zip')
data.head()


# In[3]:


# Split X and y data into 2 dataframes
X = data.loc[:, data.columns != 'label']
#X.head()
y = data.loc[:, data.columns == 'label']
#y.head()


# In[4]:


X_train,X_test,y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)
print('X_train:', len(X_train))
print('y_train:', len(y_train), '\n')
print('X_test:', len(X_test))
print('y_test:', len(y_test))


# In[5]:


svm_clf = SVC(verbose=True)
svm_clf.fit(X_train, y_train)


# In[7]:


predictions = svm_clf.predict(X_test) # this will return the class that it belongs to (e.g 1, 2, 3 or 4)
# np.argmax(prediction_scores) # Another way of seeing the highest co nfident class
print('Category 1 was predicted:', list(predictions).count(1))
print('Category 2 was predicted:', list(predictions).count(2))
print('Category 3 was predicted:', list(predictions).count(3))
print('Category 4 was predicted:', list(predictions).count(4))


# In[8]:


# See the percentage score of each class (confidence score)
predictions_proba = svm_clf.decision_function(X_test)
predictions_proba


# In[14]:


score = svm_clf.score(X_test.values, y_test.values)
cm = metrics.confusion_matrix(y_test.values, predictions)
print(cm)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


# ## Labels
# - 1 -- 0.5% up before 0.3% down
# - 2 -- neither hit 0.3% up or down
# - 3 -- 0.3% up and also 0.3% down
# - 4 -- 0.5% down before 0.3% up

# In[15]:


max_ = 4
for i in range(max_):
    print(f'Num of label {i+1}: {list(y_test.values).count(i+1)} -- Predicted: {list(predictions).count(i+1)} -- Correct Preds {round((cm[i][i]/list(predictions).count(i+1))*100, 2)}% ({cm[i][i]})')
    


# ## Trading simulation

# In[16]:


def simulation_label(y_test, initial_investment, maker, taker, confidence=None, leverage=1, strategy='longshort'):
    
    gain = 0.005
    gain = 1+ (gain * leverage)
    loss = 0.003
    loss = (1- (loss* leverage))
    maker = maker*leverage
    taker = taker*leverage
    
    balance_record = [initial_investment]
    
    for index in range(len(predictions)):  
        if predictions[index] == 1 or predictions[index] == 4:
            if confidence is None or confidence is not None and max(predictions_proba[index]) >= confidence:
                if strategy == 'long':
                    expected_label = 1
                elif strategy == 'short':
                    expected_label = 4 
                elif strategy == 'longshort':
                    expected_label = predictions[index]
                
                actual_label = int(list(y_test.values)[index])
                # actual balance update after buying, accounting for the taker fee
                buying_balance = balance_record[-1]*(1-taker)

                # if the actual label of the next candle is 4 - then reduce 0.3% or add 0.5% to the balance depending on the strategy
                if actual_label == 1:
                    if strategy == 'long' or expected_label == 1:
                        new_balance = (buying_balance*gain)*(1-maker)
                    elif strategy == 'short' or expected_label == 4:
                        new_balance = (buying_balance*loss)*(1-maker)
                    balance_record.append(new_balance)

                if actual_label == 2:
                    # WE ASSUME THAT THE COIN IS SOLD AT X% LOSS (since we have no knowledge of what the closing price is)
                    new_balance = (buying_balance*1)*(1-maker)
                    balance_record.append(new_balance)

                if actual_label == 3:
                    # WE ASSUME THAT THE COIN IS SOLD AT X% LOSS (since we have no knowledge of what the closing price is)
                    new_balance = (buying_balance*1)*(1-maker)
                    balance_record.append(new_balance)

                if actual_label == 4:
                    # if the actual label of the next candle is 4 - then reduce 0.3% or add 0.5% to the balance depending on the strategy
                    if strategy == 'long' or expected_label == 4:
                        new_balance = (buying_balance*loss)*(1-maker)
                    elif strategy == 'short' or expected_label == 1:
                        new_balance = (buying_balance*gain)*(1-maker)
                    balance_record.append(new_balance)
        
    return balance_record


# In[17]:


def plot_balance(balance_record, initial_investment,leverage=1):
    #delete the final updated balance (last index so it matches the length of the dataframe rows - because the balance record started at 1000 before any transaction happened)
    balance_record.pop(-1) 
    plt.figure(figsize=(10,10))
    plt.plot(balance_record)
    plt.xlabel("date")
    plt.ylabel("Balance")
    plt.title(f"Simulation Initial Investment {balance_record[0]}€ - Leverage {leverage}x")
    plt.show()

    print(f'Initial balance record: {initial_investment}€')
    print(f'Final balance record: {balance_record[-1]}€')


# In[18]:


maker = 0.0002 # limit order
taker = 0.0004 # market order
confidence = 0.30 # this will simulate only trades where predictions have a minimum confidence of X%.
initial_investment = 1000
leverage=1


# In[19]:


strategy = 'long' # can be 'long' or 'short'
balance_record = simulation_label(y_test, initial_investment, maker, taker, confidence, leverage, strategy)
plot_balance(balance_record, initial_investment,leverage)


# In[20]:


maker = 0.0002 # limit order
taker = 0.0004 # market order
confidence = 0.30 # this will simulate only trades where predictions have a minimum confidence of X%.
initial_investment = 1000
leverage=1


# In[21]:


strategy = 'short' # can be 'long' or 'short'
balance_record = simulation_label(y_test, initial_investment, maker, taker, confidence, leverage, strategy)
plot_balance(balance_record, initial_investment,leverage)


# In[22]:


# In this case, the model performed better with the longing strategy than shorting. This could be due
# to the fact that the test market data is mostly bullish therefore benefiting the strategy. 
# On the other hand, it could also mean that the model is better at identifying long profitable positions than short ones.


# In[30]:


maker = 0.0002 # limit order
taker = 0.0004 # market order
confidence = 0.30 # this will simulate only trades where predictions have a minimum confidence of X%.
initial_investment = 1000
leverage=1


# In[31]:


strategy = 'longshort' # can be 'long' or 'short'
balance_record = simulation_label(y_test, initial_investment, maker, taker, confidence, leverage, strategy)
plot_balance(balance_record, initial_investment,leverage)


# In[35]:


# save the model to disk
filename = 'SVM_model_1.0.sav'
pickle.dump(svm_clf, open(filename, 'wb'))
  
# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)


# In[ ]:




