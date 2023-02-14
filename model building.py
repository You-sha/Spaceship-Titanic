# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:02:11 2023

@author: Shumail
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('Cleaned and imputed data 2.csv')
df_test = pd.read_csv('Cleaned and imputed test data.csv')

df_train.columns
df_train.dtypes

features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
            'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 
            'Grouped', 'Deck', 'Side', 'Has_expenses', 'Is_Embryo']

X = pd.get_dummies(df_train[features])
y = df_train['Transported']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)

model = LogisticRegression(max_iter=10000)
model.fit(X_train,y_train)
model.score(X_test,y_test)  # 80%

y_pred_log = model.predict(pd.get_dummies(df_test[features]))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

knn = KNeighborsClassifier()
param_grid = {'n_neighbors':np.arange(2,15)}
knn_gscv = GridSearchCV(knn, param_grid, cv=5)
knn_gscv.fit(X,y)


knn2 = KNeighborsClassifier(n_neighbors=knn_gscv.best_params_['n_neighbors'])
knn2.fit(X,y)
knn2.score(X,y) # 81.4%

y_pred_knn = knn2.predict(pd.get_dummies(df_test[features]))


from sklearn.ensemble import RandomForestClassifier

fr = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
fr.fit(X,y)
fr.score(X,y) # ~77%

y_pred_rf = fr.predict(pd.get_dummies(df_test[features]))

out_test = pd.get_dummies(df_test[features])

model_out = LogisticRegression(solver='liblinear')
model_out.fit(X,y)
y_pred2 = model_out.predict(df_test)


Logist_out = pd.DataFrame({'PassengerId':df_test.PassengerId, 'Transported': y_pred_log})
Logist_out.to_csv('logist_pred.csv',index=False)

knn_out = pd.DataFrame({'PassengerId':df_test.PassengerId, 'Transported': y_pred_knn})
knn_out.to_csv('knn_pred.csv',index=False)

rf_out = pd.DataFrame({'PassengerId':df_test.PassengerId, 'Transported': y_pred_rf})
rf_out.to_csv('rf_pred.csv',index=False)












