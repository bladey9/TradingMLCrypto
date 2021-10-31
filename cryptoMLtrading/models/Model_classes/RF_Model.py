#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle


# In[5]:


class RF:
    
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
        self.model = AdaBoostClassifier(RandomForestClassifier(verbose=1), n_estimators=500,algorithm="SAMME.R", learning_rate=0.5)
        self.model.fit(self.X_train, self.y_train)
        acc = self.model.score(self.X_test.values, self.y_test.values)
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
        
    def load_model(self, directory_path):
        # load the model from disk
        model_name = f'{directory_path}.sav'
        self.model = pickle.load(open(model_name, 'rb'))
   



