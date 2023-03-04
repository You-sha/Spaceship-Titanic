# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:02:11 2023

@author: Yousha
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
            'Grouped', 'Deck', 'Has_expenses', 'Side', 'Is_Embryo']

X = pd.get_dummies(df_train[features])
y = df_train['Transported']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)

model = LogisticRegression(max_iter=10000)
model.fit(X_train,y_train)
model.score(X_test,y_test)  # 80%

model2 = LogisticRegression(max_iter=10000)
model2.fit(X,y)
model2.score(X,y)

y_pred_log = model.predict(pd.get_dummies(df_test[features]))
y_pred_log2 = model2.predict(pd.get_dummies(df_test[features]))

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

from sklearn.ensemble import GradientBoostingClassifier
gbr = GradientBoostingClassifier(random_state = 1)
  
# Fit to training set
gbr.fit(X, y)
gbr.score(X,y)
pred_y_gbr = gbr.predict(pd.get_dummies((df_test[features])))


from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X,y)
xgb.score(X,y)

y_pred_xgb = xgb.predict(pd.get_dummies((df_test[features])))

gbc = GradientBoostingClassifier()
parameters = {
    "n_estimators":[5,50,100],
    "max_depth":[1,3,5],
    "learning_rate":[0.01,0.1,1]
}

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

cv = RandomizedSearchCV(gbc, parameters, n_iter=27, scoring='accuracy', n_jobs=-1, cv=5, random_state=1)
cv.fit(X,y)
cv.best_params_

gbc1 = GradientBoostingClassifier(n_estimators=50,max_depth=5,learning_rate=0.1)

gbc1.fit(X,y)
gbc1.score(X,y)
pred_y_gbr2 = gbc1.predict(pd.get_dummies((df_test[features])))

from sklearn.metrics import plot_confusion_matrix
from pandas.plotting import scatter_matrix

plot_confusion_matrix(gbc1,X,y,cmap=plt.cm.Blues)
plt.title('Gradient Boosting Classifier')
plt.suptitle('Confusion Matrix')
plt.savefig('GBC CM.png',dpi=600)

import seaborn as sns

df_train.columns

correlations = df_train.corr()
correlations["Transported"].sort_values(ascending=False)

plt.figure(figsize=(10, 10))
sns.heatmap(
    correlations, square=True, linewidths=2, annot=True, cbar_kws={"shrink": 0.82}
)
plt.title("Correlation Matrix", fontsize=16, pad=30)
plt.show()



Logist_out = pd.DataFrame({'PassengerId':df_test.PassengerId, 'Transported': y_pred_log})
Logist_out.to_csv('logist_pred.csv',index=False)

Logist_out2 = pd.DataFrame({'PassengerId':df_test.PassengerId, 'Transported': y_pred_log2})
Logist_out2.to_csv('logist_pred2.csv',index=False)

knn_out = pd.DataFrame({'PassengerId':df_test.PassengerId, 'Transported': y_pred_knn})
knn_out.to_csv('knn_pred.csv',index=False)

rf_out = pd.DataFrame({'PassengerId':df_test.PassengerId, 'Transported': y_pred_rf})
rf_out.to_csv('rf_pred.csv',index=False)

gbr_out = pd.DataFrame({'PassengerId':df_test.PassengerId, 'Transported': pred_y_gbr})
gbr_out.to_csv('gbr_pred.csv',index=False)

xgb_out = pd.DataFrame({'PassengerId':df_test.PassengerId, 'Transported':y_pred_xgb.astype('bool')})
xgb_out.to_csv('xgb_pred.csv',index=False)

gbc_out = pd.DataFrame({'PassengerId':df_test.PassengerId, 'Transported':pred_y_gbr2})
gbc_out.to_csv('gbr_pred2.csv',index=False)



