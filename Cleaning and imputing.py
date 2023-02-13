# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 13:04:25 2023

@author: Yousha
"""
# Source: https://www.kaggle.com/code/mateuszk013/spaceship-titanic-81-eda-ml/notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df.dtypes
train_df.columns
train_df.head()
train_df.tail()
train_df.isna().any()
train_df.info()
train_df.describe(include='all')

(train_df.CryoSleep == True).sum()/len(train_df)


train_df_copy = train_df.copy()

train_df_copy['Expenses'] = train_df_copy[['RoomService', 'FoodCourt',
                                           'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)

train_df_copy.Age = train_df_copy.Age.fillna(train_df_copy.Age.median())
train_df_copy.loc[train_df_copy.Adult_spending_awake ==False, 'Age'] = 0

train_df_copy['Adult_spending_awake'] = (train_df_copy['Expenses'] > 0) & (train_df_copy['Age'] >=13) & (train_df_copy['CryoSleep'] == False)

train_df_copy['Cryosleep'] = 0
train_df_copy.loc[train_df_copy['Expenses'] == 0, 'Cryosleep'] = 1
train_df_copy.loc[train_df_copy.CryoSleep.astype('str') == 'True', 'Cryosleep'] = 1
train_df_copy.loc[train_df_copy.CryoSleep.astype('str') == 'False', 'Cryosleep'] = 0
train_df_copy['Cryosleep'] = train_df_copy['Cryosleep'].astype('bool')
train_df_copy['CryoSleep'] = train_df_copy['Cryosleep']
train_df_copy.drop('Cryosleep',axis=1,inplace=True)
train_df_copy.drop('Name',axis=1,inplace=True)

train_df_copy.loc[train_df_copy.CryoSleep == True,['RoomService', 'FoodCourt','ShoppingMall', 'Spa', 'VRDeck']] = 0
train_df_copy.loc[train_df_copy.CryoSleep == True,['RoomService', 'FoodCourt','ShoppingMall', 'Spa', 'VRDeck']].isna().sum()

train_df_copy['Age'].plot(kind='hist')
train_df_copy['Adults'] = train_df_copy['Age'] >= 13

train_df_copy['Adult_and_spending'] = (train_df_copy['Expenses'] > 0) & (train_df_copy['Age'] >=13)
train_df_copy.loc[train_df_copy.Adult_and_spending == True]

train_df_copy.RoomService = train_df_copy.RoomService.fillna(train_df_copy.RoomService.mean())
train_df_copy.loc[train_df_copy.Adult_and_spending ==False, 'RoomService'] = 0

train_df_copy.FoodCourt = train_df_copy.FoodCourt.fillna(train_df_copy.FoodCourt.mean())
train_df_copy.loc[train_df_copy.Adult_and_spending ==False, 'FoodCourt'] = 0

train_df_copy.ShoppingMall = train_df_copy.ShoppingMall.fillna(train_df_copy.ShoppingMall.mean())
train_df_copy.loc[train_df_copy.Adult_and_spending ==False, 'ShoppingMall'] = 0

train_df_copy.Spa = train_df_copy.Spa.fillna(train_df_copy.Spa.mean())
train_df_copy.loc[train_df_copy.Adult_and_spending ==False, 'Spa'] = 0

train_df_copy.VRDeck = train_df_copy.VRDeck.fillna(train_df_copy.VRDeck.mean())
train_df_copy.loc[train_df_copy.Adult_and_spending ==False, 'VRDeck'] = 0

train_df_copy.HomePlanet = train_df_copy.HomePlanet.fillna('Earth')
train_df_copy.Destination = train_df_copy.Destination.fillna('TRAPPIST-1e')
train_df_copy.VIP = train_df_copy.VIP.fillna('False')
train_df_copy.VIP = train_df_copy.VIP.astype('bool')

train_df_copy['Cabin'] = train_df_copy.Cabin.fillna(method='ffill')


train_df_copy['Group_nums'] = train_df_copy.PassengerId.apply(lambda x: x.split('_')).apply(lambda x: x[0])
train_df_copy['Grouped'] = ((train_df_copy['Group_nums'].value_counts() > 1).reindex(train_df_copy['Group_nums'])).tolist()
train_df_copy['Deck'] = train_df_copy.Cabin.apply(lambda x: str(x).split('/')).apply(lambda x: x[0])
train_df_copy['Side'] = train_df_copy.Cabin.apply(lambda x: str(x).split('/')).apply(lambda x: x[2])
train_df_copy['Has_expenses'] = train_df_copy['Expenses'] > 0
train_df_copy['Is_Embryo'] = train_df_copy['Age'] == 0


plt.bar(train_df_copy['Side'],train_df_copy['Transported'])
plt.bar(train_df_copy['Deck'],train_df_copy['Transported'])

train_df_copy.columns
train_df_copy.drop(['IsEmbryo'],axis=1, inplace=True)

train_df_copy.to_csv('Cleaned and imputed data 2.csv',index=False)












