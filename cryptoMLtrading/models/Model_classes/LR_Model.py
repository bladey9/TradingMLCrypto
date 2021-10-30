#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import pickle


# In[9]:


class LR:
    
    def __init__(self, train_data, valid_data, test_data):
        self.train = train_data
        self.valid = valid_data
        self.test = test_data
        self.run()
        
    def run(self):
        self.X_train, self.y_train = self.split(self.train)
        self.X_valid, self.y_valid = self.split(self.valid)
        self.X_test, self.y_test = self.split(self.test)
        
    def split(self, dataframe):
        # Split X and y data into 2 dataframes
        X_train = dataframe.loc[:, dataframe.columns != 'label']
        #X.head()
        y_train = dataframe.loc[:, dataframe.columns == 'label']
        #y.head()
        return X_train, y_train        
    
    def run_model(self):
        self.model = LogisticRegression(verbose=1)
        self.model.fit(self.X_train, self.y_train)
        acc = self.model.score(X_test.values, y_test.values)
        print("Accuracy on test data: ", acc)
        
    def confusion_matrix(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        score = self.model.score(X_test.values, y_test.values)
        cm = metrics.confusion_matrix(y_test.values, predictions)
        plt.figure(figsize=(9,9))
        sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        all_sample_title = 'Accuracy Score: {0}'.format(score)
        plt.title(all_sample_title, size = 15)
        plt.show()
        
        max_ = 4
        for i in range(max_):
            print(f'Num of label {i+1}: {list(y_test.values).count(i+1)} -- Predicted: {list(predictions).count(i+1)} -- True positives {round((cm[i][i]/list(predictions).count(i+1))*100, 2)}% ({cm[i][i]})')

    def save_model(self, directory_path):
        # save the model to disk
        model_name = f'{directory_path}.sav'
        pickle.dump(self.model, open(model_name, 'wb'))
        
    def load_model(self,directory_path):
        # load the model from disk
        model_name = f'{directory_path}.sav'
        self.model = pickle.load(open(model_name, 'rb'))
        
            


# In[10]:


# Loading concatenated dataframes
train= pd.read_csv('../../featureGen/PROCESSED_COINS/CONCAT_ALL_COINS/DF_5_Candles_Concat_ALL_COINS_TRAIN_WEAK_LEARNERS.csv')
valid = pd.read_csv('../../featureGen/PROCESSED_COINS/CONCAT_ALL_COINS/DF_5_Candles_Concat_ALL_COINS_VALID_WEAK_LEARNERS.csv')
test = pd.read_csv('../../featureGen/PROCESSED_COINS/CONCAT_ALL_COINS/DF_5_Candles_Concat_ALL_COINS_TEST.csv')

LR = LR(train, valid, test)
#RF.run_model()
#RF.save_model('../trained_models_2/LR_model_1_EE')

LR.load_model('../trained_models_2/LR_model_1_EE')
LR.confusion_matrix(LR.X_test, LR.y_test)


# In[ ]:




