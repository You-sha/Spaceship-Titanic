# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 18:29:43 2023

@author: Yousha
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data.dtypes
train_data.columns
features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age',
            'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 
            'VRDeck']

for i in (train_data):
    if train_data[i].dtype == 'float64':
        train_data[i] = train_data[i].fillna(train_data[i].mean())
for i in (test_data):
    if test_data[i].dtype == 'float64':
        test_data[i] = test_data[i].fillna(test_data[i].mean())
        
train_data.HomePlanet = train_data.HomePlanet.fillna('Earth')
train_data.CryoSleep = train_data.CryoSleep.fillna(train_data.CryoSleep.mean())
train_data.Destination = train_data.Destination.fillna('TRAPPIST-1e')
train_data.VIP = train_data.VIP.fillna(train_data.VIP.mean())
train_data.isna().sum()

test_data.HomePlanet = test_data.HomePlanet.fillna('Earth')
test_data.CryoSleep = test_data.CryoSleep.fillna(test_data.CryoSleep.mean())
test_data.Destination = test_data.Destination.fillna('TRAPPIST-1e')
test_data.VIP = test_data.VIP.fillna(test_data.VIP.mean())
test_data.isna().sum()

X = pd.get_dummies(train_data[features])
y = train_data['Transported']
X_test = pd.get_dummies(test_data[features])

# for i in (f_data):
#     if f_data[i].dtype == 'float64':
#         f_data[i] = f_data[i].fillna(f_data[i].mean())
# for i in (ft_data):
#     if ft_data[i].dtype == 'float64':
#         ft_data[i] = ft_data[i].fillna(ft_data[i].mean())
        
# f_data.HomePlanet = f_data.HomePlanet.fillna('Earth')
# f_data.CryoSleep = f_data.CryoSleep.fillna(f_data.CryoSleep.mean())
# f_data.Destination = f_data.Destination.fillna('TRAPPIST-1e')
# f_data.VIP = f_data.VIP.fillna(f_data.VIP.mean())
# f_data.isna().sum()

# ft_data.HomePlanet = ft_data.HomePlanet.fillna('Earth')
# ft_data.CryoSleep = ft_data.CryoSleep.fillna(ft_data.CryoSleep.mean())
# ft_data.Destination = ft_data.Destination.fillna('TRAPPIST-1e')
# ft_data.VIP = ft_data.VIP.fillna(ft_data.VIP.mean())
# ft_data.isna().sum()

# X = pd.get_dummies(f_data[0:-1])
# y = train_data['Transported']
# X_test = pd.get_dummies(ft_data)


## Models

## Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter = 1000)

model.fit(X,y)
model.score(X,y).round(2) ## 79%

y_pred = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Transported': y_pred})
output.to_csv('liblinear pred.csv',index=False)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

## K Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier()
param_grid = {"n_neighbors": np.arange(2,11)}  
knn_gscv = GridSearchCV(knn, param_grid, cv=7)  
knn_gscv.fit(X,y)

knn_final = KNeighborsClassifier(n_neighbors=knn_gscv.best_params_['n_neighbors'])
knn_final.fit(X,y)
y_pred2 = knn_final.predict(X_test)
knn_final.score(X,y).round(2)   #81%
knn_out = pd.DataFrame({'PassengerId':test_data.PassengerId, 'Transported': y_pred2})
knn_out.to_csv('knn_pred.csv',index=False)

from sklearn.ensemble import RandomForestClassifier

fr = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
fr.fit(X,y)
fr.score(X,y)   #78%
y_pred3 = fr.predict(X_test)


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(model,X,y, cmap=plt.cm.Blues)
plt.title('Spaceship Titanic Survivor Prediction')
plt.xlabel('Predicted survival')
plt.ylabel('Actual survuval')
plt.savefig('knn_confusion',dpi=600)















































