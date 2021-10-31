#!/usr/bin/env python
# coding: utf-8

# # Creating training, validation and testing data for all coins
# ## 2 type of datasets for each set are created: concatenated and sequenced

# In[1]:


import pandas as pd
import pickle


# In[2]:


class TRAIN_VALID_TEST():
    
    def __init__(self, coins):
        self.coins_dfs = coins
        self.coins_data = {}
        
    def split(self):
        for coin_pair, coin_df in self.coins_dfs.items():
            # train range 1 has the first 35% range of data
            train_range1 = int((coin_df.shape[0]/10)*3.5)
            # valid range 1 has the 35%-40% range of data (5% total)
            valid_range1 = train_range1 + int((coin_df.shape[0]/10)*0.5)

            # train range 2 has the 40%-75% range of data (35% total)
            train_range2 = valid_range1 + train_range1
            # valid range 2 has the 75%-80% range of data (5% total)
            valid_range2 = train_range2 + int((coin_df.shape[0]/10)*0.5)

            train_data1 = coin_df[:train_range1]       
            valid_data1 = coin_df[train_range1:valid_range1]
            train_data2 = coin_df[valid_range1:train_range2]
            valid_data2 = coin_df[train_range2:valid_range2]
            test_data = coin_df[valid_range2:]

            self.coins_data[coin_pair] = {'train1': train_data1, 'valid1': valid_data1, 'train2': train_data2, 'valid2': valid_data2, 'test': test_data}
            
    def run(self):
        self.split()
        train_dfs1 = []
        valid_dfs1 = []
        train_dfs2 = []
        valid_dfs2 = []
        test_dfs = []
                
        for coin_pair, data_type in self.coins_data.items():
            train_dfs1.append(self.coins_data[coin_pair]['train1'])
            valid_dfs1.append(self.coins_data[coin_pair]['valid1'])
            train_dfs2.append(self.coins_data[coin_pair]['train2'])
            valid_dfs2.append(self.coins_data[coin_pair]['valid2'])
            test_dfs.append(self.coins_data[coin_pair]['test'])
        
        # Concatenate dataframes from different coins together
        train1 = pd.concat(train_dfs1, ignore_index=True)
        valid1 = pd.concat(valid_dfs1, ignore_index=True)
        train2 = pd.concat(train_dfs2, ignore_index=True)
        valid2 = pd.concat(valid_dfs2, ignore_index=True)
        test = pd.concat(test_dfs, ignore_index=True)

        # Shuffle dataframes' rows
        train1 = train1.sample(frac = 1, random_state=42)
        valid1 = valid1.sample(frac = 1, random_state=42)
        train2 = train2.sample(frac = 1, random_state=42)
        valid2 = valid2.sample(frac = 1, random_state=42)
        test = test.sample(frac = 1, random_state=42)
        
        return train1, valid1, train2, valid2, test


# ### How to use TRAIN_VALID_TEST() class

# In[3]:


#dotusdt_df = pd.read_csv('PROCESSED_COINS/Concat/DF_5_Candles_Concat_DOTUSDT.csv.zip')
#ftmusdt_df = pd.read_csv('PROCESSED_COINS/Concat/DF_5_Candles_Concat_FTMUSDT.csv.zip')
#icpusdt_df = pd.read_csv('PROCESSED_COINS/Concat/DF_5_Candles_Concat_ICPUSDT.csv.zip')
#maticusdt_df = pd.read_csv('PROCESSED_COINS/Concat/DF_5_Candles_Concat_MATICUSDT.csv.zip')
#omgusdt_df = pd.read_csv('PROCESSED_COINS/Concat/DF_5_Candles_Concat_OMGUSDT.csv.zip')
#solusdt_df = pd.read_csv('PROCESSED_COINS/Concat/DF_5_Candles_Concat_SOLUSDT.csv.zip')


# In[16]:


#dotusdt_df = pd.read_csv('PROCESSED_COINS/Sequenced/DF_sequence_DOTUSDT.csv')[5:]
#ftmusdt_df = pd.read_csv('PROCESSED_COINS/Sequenced/DF_sequence_FTMUSDT.csv')[5:]
#icpusdt_df = pd.read_csv('PROCESSED_COINS/Sequenced/DF_sequence_ICPUSDT.csv')[5:]
#maticusdt_df = pd.read_csv('PROCESSED_COINS/Sequenced/DF_sequence_MATICUSDT.csv')[5:]
#omgusdt_df = pd.read_csv('PROCESSED_COINS/Sequenced/DF_sequence_OMGUSDT.csv')[5:]
#solusdt_df = pd.read_csv('PROCESSED_COINS/Sequenced/DF_sequence_SOLUSDT.csv')[5:]


# In[17]:


#coins_dfs = {'DOTUSDT':dotusdt_df, 'FTMUSDT':ftmusdt_df, 'ICPUSDT':icpusdt_df, 'MATICUSDT':maticusdt_df, 'OMGUSDT':omgusdt_df, 'SOLUSDT':solusdt_df}
#data = TRAIN_VALID_TEST(coins_dfs)
#train1, valid1, train2, valid2, test = data.run()


# ### How to save dataframes

# In[18]:


# Saving concatenated dataframes
#train1.to_csv('PROCESSED_COINS/CONCAT_ALL_COINS/DF_5_Candles_Concat_ALL_COINS_TRAIN_WEAK_LEARNERS.csv',index=False)
#valid1.to_csv('PROCESSED_COINS/CONCAT_ALL_COINS/DF_5_Candles_Concat_ALL_COINS_VALID_WEAK_LEARNERS.csv', index=False)
#train2.to_csv('PROCESSED_COINS/CONCAT_ALL_COINS/DF_5_Candles_Concat_ALL_COINS_TRAIN_ENSEMBLE.csv',index=False)
#valid2.to_csv('PROCESSED_COINS/CONCAT_ALL_COINS/DF_5_Candles_Concat_ALL_COINS_VALID_ENSEMBLE.csv', index=False)
#test.to_csv('PROCESSED_COINS/CONCAT_ALL_COINS/DF_5_Candles_Concat_ALL_COINS_TEST.csv', index=False)


# In[19]:


# Saving sequenced dataframes
#train1.to_csv('PROCESSED_COINS/Sequenced_ALL_COINS/DF_5_Candles_Sequenced_ALL_COINS_TRAIN_WEAK_LEARNERS.csv',index=False)
#valid1.to_csv('PROCESSED_COINS/Sequenced_ALL_COINS/DF_5_Candles_Sequenced_ALL_COINS_VALID_WEAK_LEARNERS.csv',index=False)
#train2.to_csv('PROCESSED_COINS/Sequenced_ALL_COINS/DF_5_Candles_Sequenced_ALL_COINS_TRAIN_ENSEMBLE.csv',index=False)
#valid2.to_csv('PROCESSED_COINS/Sequenced_ALL_COINS/DF_5_Candles_Sequenced_ALL_COINS_VALID_ENSEMBLE.csv', index=False)
#test.to_csv('PROCESSED_COINS/Sequenced_ALL_COINS/DF_5_Candles_Sequenced_ALL_COINS_TEST.csv', index=False)


# ### How to load dataframes

# In[20]:


# Loading concatenated dataframes
#train_concat1 = pd.read_csv('PROCESSED_COINS/CONCAT_ALL_COINS/DF_5_Candles_Concat_ALL_COINS_TRAIN_WEAK_LEARNERS.csv')
#valid_concat1 = pd.read_csv('PROCESSED_COINS/CONCAT_ALL_COINS/DF_5_Candles_Concat_ALL_COINS_VALID_WEAK_LEARNERS.csv')
#train_concat2 = pd.read_csv('PROCESSED_COINS/CONCAT_ALL_COINS/DF_5_Candles_Concat_ALL_COINS_TRAIN_ENSEMBLE.csv')
#valid_concat2 = pd.read_csv('PROCESSED_COINS/CONCAT_ALL_COINS/DF_5_Candles_Concat_ALL_COINS_VALID_ENSEMBLE.csv')
#test_concat = pd.read_csv('PROCESSED_COINS/CONCAT_ALL_COINS/DF_5_Candles_Concat_ALL_COINS_TEST.csv')


# In[21]:


# Loading sequenced dataframes
#train_sequenced1 = pd.read_csv('PROCESSED_COINS/Sequenced_ALL_COINS/DF_5_Candles_Sequenced_ALL_COINS_TRAIN_WEAK_LEARNERS.csv')
#valid_sequenced1 = pd.read_csv('PROCESSED_COINS/Sequenced_ALL_COINS/DF_5_Candles_Sequenced_ALL_COINS_VALID_WEAK_LEARNERS.csv')
#train_sequenced2 = pd.read_csv('PROCESSED_COINS/Sequenced_ALL_COINS/DF_5_Candles_Sequenced_ALL_COINS_TRAIN_ENSEMBLE.csv')
#valid_sequenced2 = pd.read_csv('PROCESSED_COINS/Sequenced_ALL_COINS/DF_5_Candles_Sequenced_ALL_COINS_VALID_ENSEMBLE.csv')
#test_sequenced = pd.read_csv('PROCESSED_COINS/Sequenced_ALL_COINS/DF_5_Candles_Sequenced_ALL_COINS_TEST.csv')

