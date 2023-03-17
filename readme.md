# <p align="center"> Spaceship Titanic </p>

<p align="center"><b>Tools used</b>: Python (<b>Numpy, Pandas, Scikit-learn, Matplotlib</b>)</p>
    
<p align="center"><b>Content</b>: Exploratory data analysis, imputing null values, feature engineering, model building and tuning.</p>

<p align="center"> <b>Kaggle Notebook: https://www.kaggle.com/code/youusha/spaceship-titanic-80-5-data-imputing-focus</b></p>
    
---

**Notes:**

**This is what I did to get an 80.5% accuracy on my Spaceship Titanic competition submission**, as pretty much a beginner. But it is far from perfect and I would really appreciate any constructive feedback on this project.

Since I knew extremely little about all the machine learning models and hyperparameters and what-not when I was working on this, I just decided to follow the basics and do my best to fill null values as accurately as possible. Then I just trained and predicted using the basic models that I knew, and a couple I know nothing about (haha dw I'll learn them soon).

If you wish, you can directly use the **final dataset** that I created in this notebook and used for prediction [here](https://www.kaggle.com/datasets/youusha/spaceship-titanic-cleaned-and-imputed).

And [this](https://www.kaggle.com/code/mateuszk013/spaceship-titanic-81-eda-ml/notebookhttps://www.kaggle.com/code/mateuszk013/spaceship-titanic-81-eda-ml/notebook) is the notebook in this competition that helped me a lot on this project. Basically gave me a sense on how to view data, and how to create more from what I have. **The new features that I use are the ones created in this notebook**.

With all that out of the way, let's get started :)

---

<h2><b>Table of contents:</b></h2>

* [Exploratory Data Analysis](https://github.com/You-sha/Spaceship-Titanic#exploratory-data-analysis)

* [Imputing Null Values](https://github.com/You-sha/Spaceship-Titanic#imputing-null-values)

* [Feature Engineering](https://github.com/You-sha/Spaceship-Titanic#feature-engineering)

* [Model Building](https://github.com/You-sha/Spaceship-Titanic#model-building)

* [Results](https://github.com/You-sha/Spaceship-Titanic#-results-) 

---

# Exploratory Data Analysis

First, we are going to import the libraries and modules we will be using:


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

And the datasets:


```python
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
```

---

Let's take a look at the data:


```python
train_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>HomePlanet</th>
      <th>CryoSleep</th>
      <th>Cabin</th>
      <th>Destination</th>
      <th>Age</th>
      <th>VIP</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
      <th>Name</th>
      <th>Transported</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0001_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>B/0/P</td>
      <td>TRAPPIST-1e</td>
      <td>39.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Maham Ofracculy</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>24.0</td>
      <td>False</td>
      <td>109.0</td>
      <td>9.0</td>
      <td>25.0</td>
      <td>549.0</td>
      <td>44.0</td>
      <td>Juanna Vines</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0003_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>58.0</td>
      <td>True</td>
      <td>43.0</td>
      <td>3576.0</td>
      <td>0.0</td>
      <td>6715.0</td>
      <td>49.0</td>
      <td>Altark Susent</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0003_02</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>33.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>1283.0</td>
      <td>371.0</td>
      <td>3329.0</td>
      <td>193.0</td>
      <td>Solam Susent</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0004_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/1/S</td>
      <td>TRAPPIST-1e</td>
      <td>16.0</td>
      <td>False</td>
      <td>303.0</td>
      <td>70.0</td>
      <td>151.0</td>
      <td>565.0</td>
      <td>2.0</td>
      <td>Willy Santantines</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df.isna().any()
```




    PassengerId     False
    HomePlanet       True
    CryoSleep        True
    Cabin            True
    Destination      True
    Age              True
    VIP              True
    RoomService      True
    FoodCourt        True
    ShoppingMall     True
    Spa              True
    VRDeck           True
    Name             True
    Transported     False
    dtype: bool



**Observation:** All the columns have null values, except for ```PassengerId``` and ```Transported```.

Let's look at exactly how many nulls we have:


```python
print('Sum of nulls:')
train_df.isna().sum()
```

    Sum of nulls:
    




    PassengerId       0
    HomePlanet      201
    CryoSleep       217
    Cabin           199
    Destination     182
    Age             179
    VIP             203
    RoomService     181
    FoodCourt       183
    ShoppingMall    208
    Spa             183
    VRDeck          188
    Name            200
    Transported       0
    dtype: int64



So, the first thing we have to do is start filling these null values, or the ml models won't work. The first thing I want to do is figure out a way to fill up the ```CryoSleep``` values.

---

# Imputing Null Values

I am much more comfortable with doing all this on a new copy, just in case I mess up.


```python
train_df_copy = train_df.copy()
```

Now I want temporarily to make an ```Expenses``` column. I'm making this column because **if someone is in cryosleep, they are not spending any money**. So, knowing someone's expenses can help us impute values for ```CryoSleep```.


```python
train_df_copy['Expenses'] = train_df_copy[['RoomService', 'FoodCourt',
                                           'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
```

Since we can't really guess anyone's age, I'll just impute these null values with the median. An interesting thing I observed is that **only people who are 13+ have expenses**. I guess the little kids don't get pocket money. Quite unfair.


```python
train_df_copy.Age = train_df_copy.Age.fillna(train_df_copy.Age.median())
```

Let's take a look at our dataset now:


```python
train_df_copy.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>HomePlanet</th>
      <th>CryoSleep</th>
      <th>Cabin</th>
      <th>Destination</th>
      <th>Age</th>
      <th>VIP</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
      <th>Name</th>
      <th>Transported</th>
      <th>Expenses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0001_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>B/0/P</td>
      <td>TRAPPIST-1e</td>
      <td>39.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Maham Ofracculy</td>
      <td>False</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>24.0</td>
      <td>False</td>
      <td>109.0</td>
      <td>9.0</td>
      <td>25.0</td>
      <td>549.0</td>
      <td>44.0</td>
      <td>Juanna Vines</td>
      <td>True</td>
      <td>736.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0003_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>58.0</td>
      <td>True</td>
      <td>43.0</td>
      <td>3576.0</td>
      <td>0.0</td>
      <td>6715.0</td>
      <td>49.0</td>
      <td>Altark Susent</td>
      <td>False</td>
      <td>10383.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0003_02</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>33.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>1283.0</td>
      <td>371.0</td>
      <td>3329.0</td>
      <td>193.0</td>
      <td>Solam Susent</td>
      <td>False</td>
      <td>5176.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0004_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/1/S</td>
      <td>TRAPPIST-1e</td>
      <td>16.0</td>
      <td>False</td>
      <td>303.0</td>
      <td>70.0</td>
      <td>151.0</td>
      <td>565.0</td>
      <td>2.0</td>
      <td>Willy Santantines</td>
      <td>True</td>
      <td>1091.0</td>
    </tr>
  </tbody>
</table>
</div>



We can see our newly created columns after ```Transported```. Funnily enough, the first person on the dataset, **Maham**, didn't spend **any** money, and they weren't even in cryosleep. Maybe they are broke? In that case I can relate with them.

---

This is where the real fun begins. First, we're going to make a new column for cryosleep, with **all** values equal to **False** (or 0):


```python
train_df_copy['Cryosleep'] = 0
```

Now, for every row where ```Expenses``` is ```0```, we're going to put ``1`` as the value. Because **if someone has not spent any money, they are proably in cryosleep**. But don't worry, we'll deal with the exceptions, like Maham, later.


```python
train_df_copy.loc[train_df_copy['Expenses'] == 0, 'Cryosleep'] = 1
```

Now, we are going to set this column's value to ``1`` wherever the original ```CryoSleep``` is equal to **True**.


```python
train_df_copy.loc[train_df_copy.CryoSleep.astype('str') == 'True', 'Cryosleep'] = 1
```

Conversely, we will put it to ``0`` wherever ``CryoSleep`` is **False**. 


```python
train_df_copy.loc[train_df_copy.CryoSleep.astype('str') == 'False', 'Cryosleep'] = 0
```

Let's take a look at this new column now:


```python
train_df_copy.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>HomePlanet</th>
      <th>CryoSleep</th>
      <th>Cabin</th>
      <th>Destination</th>
      <th>Age</th>
      <th>VIP</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
      <th>Name</th>
      <th>Transported</th>
      <th>Expenses</th>
      <th>Cryosleep</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0001_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>B/0/P</td>
      <td>TRAPPIST-1e</td>
      <td>39.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Maham Ofracculy</td>
      <td>False</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>24.0</td>
      <td>False</td>
      <td>109.0</td>
      <td>9.0</td>
      <td>25.0</td>
      <td>549.0</td>
      <td>44.0</td>
      <td>Juanna Vines</td>
      <td>True</td>
      <td>736.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0003_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>58.0</td>
      <td>True</td>
      <td>43.0</td>
      <td>3576.0</td>
      <td>0.0</td>
      <td>6715.0</td>
      <td>49.0</td>
      <td>Altark Susent</td>
      <td>False</td>
      <td>10383.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0003_02</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>33.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>1283.0</td>
      <td>371.0</td>
      <td>3329.0</td>
      <td>193.0</td>
      <td>Solam Susent</td>
      <td>False</td>
      <td>5176.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0004_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/1/S</td>
      <td>TRAPPIST-1e</td>
      <td>16.0</td>
      <td>False</td>
      <td>303.0</td>
      <td>70.0</td>
      <td>151.0</td>
      <td>565.0</td>
      <td>2.0</td>
      <td>Willy Santantines</td>
      <td>True</td>
      <td>1091.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



What we have done here is:

* First, we set **all** values for cryosleep as false.
* Next, we set cryosleep as true for **everyone who hasn't spent any money**.
* Finally, we used the original `Cryosleep colum`, to correct cryosleep status for the people who **haven't spent any money, but aren't in cryosleep**. Just in case our last step **incorrectly** classified them as **being in cryosleep**.

Logical, right?

Now, let's just replace the original column with this one. There's probably a better way of doing this than how I did it here here:


```python
train_df_copy['Cryosleep'] = train_df_copy['Cryosleep'].astype('bool')
train_df_copy['CryoSleep'] = train_df_copy['Cryosleep']
train_df_copy.drop('Cryosleep',axis=1,inplace=True)
```

Let's take another look at our dataset now:


```python
train_df_copy.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>HomePlanet</th>
      <th>CryoSleep</th>
      <th>Cabin</th>
      <th>Destination</th>
      <th>Age</th>
      <th>VIP</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
      <th>Name</th>
      <th>Transported</th>
      <th>Expenses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0001_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>B/0/P</td>
      <td>TRAPPIST-1e</td>
      <td>39.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Maham Ofracculy</td>
      <td>False</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>24.0</td>
      <td>False</td>
      <td>109.0</td>
      <td>9.0</td>
      <td>25.0</td>
      <td>549.0</td>
      <td>44.0</td>
      <td>Juanna Vines</td>
      <td>True</td>
      <td>736.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0003_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>58.0</td>
      <td>True</td>
      <td>43.0</td>
      <td>3576.0</td>
      <td>0.0</td>
      <td>6715.0</td>
      <td>49.0</td>
      <td>Altark Susent</td>
      <td>False</td>
      <td>10383.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0003_02</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>33.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>1283.0</td>
      <td>371.0</td>
      <td>3329.0</td>
      <td>193.0</td>
      <td>Solam Susent</td>
      <td>False</td>
      <td>5176.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0004_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/1/S</td>
      <td>TRAPPIST-1e</td>
      <td>16.0</td>
      <td>False</td>
      <td>303.0</td>
      <td>70.0</td>
      <td>151.0</td>
      <td>565.0</td>
      <td>2.0</td>
      <td>Willy Santantines</td>
      <td>True</td>
      <td>1091.0</td>
    </tr>
  </tbody>
</table>
</div>



**We have now replaced the values of our original `CryoSleep` column, that had missing values, with the values of our newly created `Cryosleep` column which doesn't have any null values. Then we dropped our new column.**

Our new column also states accurately that Maham is not in cryosleep, and he still hasn't spent any money on amenities, i.e., `RoomService`,`FoodCourt`,`ShoppingMall`,`Spa` and `VRDeck`.

The new column shouldn't have any null values now. Let's check just in case:


```python
train_df_copy.CryoSleep.isnull().any()
```




    False



Since the only important person in this dataset is Maham, we don't need the names column. (Or maybe we actually do and can use to to further improve prediction, but I'm just not good enough to figure out how  to do that yet.)


```python
train_df_copy.drop('Name',axis=1,inplace=True)
```

Now for the amenities, we can easily impute null values for ```Cryosleep``` == True, since **we know they are going to be zero as the person is in cryosleep**.


```python
train_df_copy.loc[train_df_copy.CryoSleep == True,['RoomService', 'FoodCourt','ShoppingMall', 'Spa', 'VRDeck']] = 0
train_df_copy.loc[train_df_copy.CryoSleep == True,['RoomService', 'FoodCourt','ShoppingMall', 'Spa', 'VRDeck']].isna().sum()
```




    RoomService     0
    FoodCourt       0
    ShoppingMall    0
    Spa             0
    VRDeck          0
    dtype: int64



Before dealing with the rest of the amenities' values, let's make some more new columns to aid us.


```python
train_df_copy['Adults'] = train_df_copy['Age'] >= 13
```

I know 13 year olds aren't adults, okay. What I mean is that **they are able to spend money at this age**. Unike a certain someone we know of. If someone has any spare change, do let me know. 

I'm not picking on Maham, I just want him to enjoy his journey on the spaceship titanic to the absolute fullest, especially with the tragedy that happens. To be honest, I am extremely happy that he didn't get transported into who-knows-what dimension. He is still with us, and we are all grateful for that, I am sure.

Jokes aside, let's make a column now that tells us if **someone is 13+** and **is spending money**.


```python
train_df_copy['Adult_and_spending'] = (train_df_copy['Expenses'] > 0) & (train_df_copy['Age'] >=13)
```

Let's take a look at the rows that are **True** for our new `Adult_and_spending` column:


```python
train_df_copy.loc[train_df_copy.Adult_and_spending == True]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>HomePlanet</th>
      <th>CryoSleep</th>
      <th>Cabin</th>
      <th>Destination</th>
      <th>Age</th>
      <th>VIP</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
      <th>Transported</th>
      <th>Expenses</th>
      <th>Adults</th>
      <th>Adult_and_spending</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0002_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>24.0</td>
      <td>False</td>
      <td>109.0</td>
      <td>9.0</td>
      <td>25.0</td>
      <td>549.0</td>
      <td>44.0</td>
      <td>True</td>
      <td>736.0</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0003_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>58.0</td>
      <td>True</td>
      <td>43.0</td>
      <td>3576.0</td>
      <td>0.0</td>
      <td>6715.0</td>
      <td>49.0</td>
      <td>False</td>
      <td>10383.0</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0003_02</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>33.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>1283.0</td>
      <td>371.0</td>
      <td>3329.0</td>
      <td>193.0</td>
      <td>False</td>
      <td>5176.0</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0004_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/1/S</td>
      <td>TRAPPIST-1e</td>
      <td>16.0</td>
      <td>False</td>
      <td>303.0</td>
      <td>70.0</td>
      <td>151.0</td>
      <td>565.0</td>
      <td>2.0</td>
      <td>True</td>
      <td>1091.0</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0005_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/0/P</td>
      <td>PSO J318.5-22</td>
      <td>44.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>483.0</td>
      <td>0.0</td>
      <td>291.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>774.0</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8687</th>
      <td>9275_03</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/97/P</td>
      <td>TRAPPIST-1e</td>
      <td>30.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>3208.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>330.0</td>
      <td>True</td>
      <td>3540.0</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8688</th>
      <td>9276_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/98/P</td>
      <td>55 Cancri e</td>
      <td>41.0</td>
      <td>True</td>
      <td>0.0</td>
      <td>6819.0</td>
      <td>0.0</td>
      <td>1643.0</td>
      <td>74.0</td>
      <td>False</td>
      <td>8536.0</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8690</th>
      <td>9279_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>G/1500/S</td>
      <td>TRAPPIST-1e</td>
      <td>26.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1872.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>True</td>
      <td>1873.0</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8691</th>
      <td>9280_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>E/608/S</td>
      <td>55 Cancri e</td>
      <td>32.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>1049.0</td>
      <td>0.0</td>
      <td>353.0</td>
      <td>3235.0</td>
      <td>False</td>
      <td>4637.0</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8692</th>
      <td>9280_02</td>
      <td>Europa</td>
      <td>False</td>
      <td>E/608/S</td>
      <td>TRAPPIST-1e</td>
      <td>44.0</td>
      <td>False</td>
      <td>126.0</td>
      <td>4688.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>True</td>
      <td>4826.0</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>5040 rows × 16 columns</p>
</div>



So there are **5040** people who **are 13+** and **are spending money**.

Now we are going to impute the values for our amenities.

We know if someone is **not an adult** and has **zero expenses**, they are either below 13, which means they **definitely** haven't spent on **any** amenities, or they are **in cryosleep**, which again means they **definitely** haven't spent on amenities.

So, wherever we have `Adult_and_spending` == False, we'll impute them with `0`.


```python
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
```

Neat. Now we are done with imputing these columns as well.

Let's take a look:


```python
train_df_copy[['RoomService', 'FoodCourt','ShoppingMall', 'Spa', 'VRDeck']].isna().sum()
```




    RoomService     0
    FoodCourt       0
    ShoppingMall    0
    Spa             0
    VRDeck          0
    dtype: int64



Perfect.

For the remaining columns, we can't figure out what values to fill in this manner. So we are just going to fill them with the values that the majority of people have in the dataset, i.e., the **mode**. 


```python
train_df_copy.HomePlanet.mode()
```




    0    Earth
    Name: HomePlanet, dtype: object




```python
train_df_copy.Destination.mode()
```




    0    TRAPPIST-1e
    Name: Destination, dtype: object




```python
train_df_copy.VIP.mode()
```




    0    False
    Name: VIP, dtype: object



So, these are the values we will be imputing with.


```python
train_df_copy.HomePlanet = train_df_copy.HomePlanet.fillna('Earth')
train_df_copy.Destination = train_df_copy.Destination.fillna('TRAPPIST-1e')
train_df_copy.VIP = train_df_copy.VIP.fillna('False')
train_df_copy.VIP = train_df_copy.VIP.astype('bool')
```

Aaand done!

Let's see how much we are done:


```python
train_df_copy.isnull().sum()
```




    PassengerId             0
    HomePlanet              0
    CryoSleep               0
    Cabin                 199
    Destination             0
    Age                     0
    VIP                     0
    RoomService             0
    FoodCourt               0
    ShoppingMall            0
    Spa                     0
    VRDeck                  0
    Transported             0
    Expenses                0
    Adults                  0
    Adult_and_spending      0
    dtype: int64



The cabin is the only column that remains with null values! 

Filling this is not easy due to my limited skill. I am just going to use **ffill** to fill these null values. What that does is basically use the **previous** value to impute the **missing** one. 

So, for example, if we have a dataset like:

[1, 2, 3, **null**, 4]

If we use **ffill** on this, it'll become:

[1, 2, 3, **3**, 4].


```python
train_df_copy['Cabin'] = train_df_copy.Cabin.fillna(method='ffill')
```


```python
train_df_copy.isnull().sum()
```




    PassengerId           0
    HomePlanet            0
    CryoSleep             0
    Cabin                 0
    Destination           0
    Age                   0
    VIP                   0
    RoomService           0
    FoodCourt             0
    ShoppingMall          0
    Spa                   0
    VRDeck                0
    Transported           0
    Expenses              0
    Adults                0
    Adult_and_spending    0
    dtype: int64



And so, we are done with imputing. Time to move on to feature engineering.

# Feature Engineering

These are the features that I am going to add to this dataset (again, I got the idea for them [here](https://www.kaggle.com/code/mateuszk013/spaceship-titanic-81-eda-ml/notebook)).


```python
train_df_copy['Group_nums'] = train_df_copy.PassengerId.apply(lambda x: x.split('_')).apply(lambda x: x[0])
train_df_copy['Grouped'] = ((train_df_copy['Group_nums'].value_counts() > 1).reindex(train_df_copy['Group_nums'])).tolist()
train_df_copy['Deck'] = train_df_copy.Cabin.apply(lambda x: str(x).split('/')).apply(lambda x: x[0])
train_df_copy['Side'] = train_df_copy.Cabin.apply(lambda x: str(x).split('/')).apply(lambda x: x[2])
train_df_copy['Has_expenses'] = train_df_copy['Expenses'] > 0
train_df_copy['Is_Embryo'] = train_df_copy['Age'] == 0
```

These specifiy:

* If someone was **alone** or **in a group**.
* Which **deck** someone was in.
* Which side (**Starboard** or **Port**).
* If the passenger was **0 years old** (i.e, an **embryo**).

Let's get rid of our temporary columns:


```python
train_df_copy.drop(['Adult_and_spending','Group_nums','Expenses'],axis=1,\
                   inplace=True)
```

This is our final dataset:


```python
train_df_copy.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>HomePlanet</th>
      <th>CryoSleep</th>
      <th>Cabin</th>
      <th>Destination</th>
      <th>Age</th>
      <th>VIP</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
      <th>Transported</th>
      <th>Adults</th>
      <th>Grouped</th>
      <th>Deck</th>
      <th>Side</th>
      <th>Has_expenses</th>
      <th>Is_Embryo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0001_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>B/0/P</td>
      <td>TRAPPIST-1e</td>
      <td>39.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>B</td>
      <td>P</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>24.0</td>
      <td>False</td>
      <td>109.0</td>
      <td>9.0</td>
      <td>25.0</td>
      <td>549.0</td>
      <td>44.0</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>F</td>
      <td>S</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0003_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>58.0</td>
      <td>True</td>
      <td>43.0</td>
      <td>3576.0</td>
      <td>0.0</td>
      <td>6715.0</td>
      <td>49.0</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>A</td>
      <td>S</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0003_02</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>33.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>1283.0</td>
      <td>371.0</td>
      <td>3329.0</td>
      <td>193.0</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>A</td>
      <td>S</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0004_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/1/S</td>
      <td>TRAPPIST-1e</td>
      <td>16.0</td>
      <td>False</td>
      <td>303.0</td>
      <td>70.0</td>
      <td>151.0</td>
      <td>565.0</td>
      <td>2.0</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>F</td>
      <td>S</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



Saving it just in case.


```python
train_df_copy.to_csv('Cleaned and imputed data.csv',index=False)
```

---

Since even our test data has missing values, we have to do **all** that to our test data as well.

### Test Data


```python
test_df_copy = test_df.copy()

test_df_copy['Expenses'] = test_df_copy[['RoomService', 'FoodCourt',
                                           'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)

test_df_copy.Age = test_df_copy.Age.fillna(test_df_copy.Age.median())

test_df_copy['Adult_spending_awake'] = (test_df_copy['Expenses'] > 0)\
                                     & (test_df_copy['Age'] >= 13)\
                                     & (test_df_copy['CryoSleep'] == False)

test_df_copy['Cryosleep'] = 0
test_df_copy.loc[test_df_copy['Expenses'] == 0, 'Cryosleep'] = 1
test_df_copy.loc[test_df_copy.CryoSleep.astype('str') == 'True', 'Cryosleep'] = 1
test_df_copy.loc[test_df_copy.CryoSleep.astype('str') == 'False', 'Cryosleep'] = 0
test_df_copy['Cryosleep'] = test_df_copy['Cryosleep'].astype('bool')
test_df_copy['CryoSleep'] = test_df_copy['Cryosleep']
test_df_copy.drop('Cryosleep',axis=1,inplace=True)
test_df_copy.drop('Name',axis=1,inplace=True)

test_df_copy.loc[test_df_copy.CryoSleep == True,['RoomService', 'FoodCourt','ShoppingMall', 'Spa', 'VRDeck']] = 0

test_df_copy['Adults'] = test_df_copy['Age'] >= 13

test_df_copy['Adult_and_spending'] = (test_df_copy['Expenses'] > 0) & (test_df_copy['Age'] >=13)
test_df_copy.loc[test_df_copy.Adult_and_spending == True]

test_df_copy.RoomService = test_df_copy.RoomService.fillna(test_df_copy.RoomService.mean())
test_df_copy.loc[test_df_copy.Adult_and_spending ==False, 'RoomService'] = 0

test_df_copy.FoodCourt = test_df_copy.FoodCourt.fillna(test_df_copy.FoodCourt.mean())
test_df_copy.loc[test_df_copy.Adult_and_spending ==False, 'FoodCourt'] = 0

test_df_copy.ShoppingMall = test_df_copy.ShoppingMall.fillna(test_df_copy.ShoppingMall.mean())
test_df_copy.loc[test_df_copy.Adult_and_spending ==False, 'ShoppingMall'] = 0

test_df_copy.Spa = test_df_copy.Spa.fillna(test_df_copy.Spa.mean())
test_df_copy.loc[test_df_copy.Adult_and_spending ==False, 'Spa'] = 0

test_df_copy.VRDeck = test_df_copy.VRDeck.fillna(test_df_copy.VRDeck.mean())
test_df_copy.loc[test_df_copy.Adult_and_spending ==False, 'VRDeck'] = 0

test_df_copy.HomePlanet = test_df_copy.HomePlanet.fillna('Earth')
test_df_copy.Destination = test_df_copy.Destination.fillna('TRAPPIST-1e')
test_df_copy.VIP = test_df_copy.VIP.fillna('False')
test_df_copy.VIP = test_df_copy.VIP.astype('bool')

test_df_copy['Cabin'] = test_df_copy.Cabin.fillna(method='ffill')

test_df_copy['Group_nums'] = test_df_copy.PassengerId.apply(lambda x: x.split('_')).apply(lambda x: x[0])
test_df_copy['Grouped'] = ((test_df_copy['Group_nums'].value_counts() > 1).reindex(test_df_copy['Group_nums'])).tolist()
test_df_copy['Deck'] = test_df_copy.Cabin.apply(lambda x: str(x).split('/')).apply(lambda x: x[0])
test_df_copy['Side'] = test_df_copy.Cabin.apply(lambda x: str(x).split('/')).apply(lambda x: x[2])
test_df_copy['Has_expenses'] = test_df_copy['Expenses'] > 0
test_df_copy['Is_Embryo'] = test_df_copy['Age'] == 0

test_df_copy.columns
test_df_copy.drop(['Expenses', 'Adult_spending_awake', 'Adult_and_spending','Adults'],axis=1, inplace=True)

test_df_copy.to_csv('Cleaned and imputed test data.csv',index=False)
```

Simple enough.

Time to build some models.

# Model Building

Let's import **Logistic Regression**. I'm also going to import **train-test split**, just for some light evaluation.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
```

Now, we import the csv's that we saved earlier.


```python
df_train = pd.read_csv('Cleaned and imputed data.csv')
df_test = pd.read_csv('Cleaned and imputed test data.csv')
```


```python
df_train.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>HomePlanet</th>
      <th>CryoSleep</th>
      <th>Cabin</th>
      <th>Destination</th>
      <th>Age</th>
      <th>VIP</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
      <th>Transported</th>
      <th>Adults</th>
      <th>Grouped</th>
      <th>Deck</th>
      <th>Side</th>
      <th>Has_expenses</th>
      <th>Is_Embryo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0001_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>B/0/P</td>
      <td>TRAPPIST-1e</td>
      <td>39.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>B</td>
      <td>P</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>24.0</td>
      <td>False</td>
      <td>109.0</td>
      <td>9.0</td>
      <td>25.0</td>
      <td>549.0</td>
      <td>44.0</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>F</td>
      <td>S</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0003_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>58.0</td>
      <td>True</td>
      <td>43.0</td>
      <td>3576.0</td>
      <td>0.0</td>
      <td>6715.0</td>
      <td>49.0</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>A</td>
      <td>S</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0003_02</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>33.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>1283.0</td>
      <td>371.0</td>
      <td>3329.0</td>
      <td>193.0</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>A</td>
      <td>S</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0004_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/1/S</td>
      <td>TRAPPIST-1e</td>
      <td>16.0</td>
      <td>False</td>
      <td>303.0</td>
      <td>70.0</td>
      <td>151.0</td>
      <td>565.0</td>
      <td>2.0</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>F</td>
      <td>S</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_test.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>HomePlanet</th>
      <th>CryoSleep</th>
      <th>Cabin</th>
      <th>Destination</th>
      <th>Age</th>
      <th>VIP</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
      <th>Group_nums</th>
      <th>Grouped</th>
      <th>Deck</th>
      <th>Side</th>
      <th>Has_expenses</th>
      <th>Is_Embryo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0013_01</td>
      <td>Earth</td>
      <td>True</td>
      <td>G/3/S</td>
      <td>TRAPPIST-1e</td>
      <td>27.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13</td>
      <td>False</td>
      <td>G</td>
      <td>S</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0018_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/4/S</td>
      <td>TRAPPIST-1e</td>
      <td>19.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>2823.0</td>
      <td>0.0</td>
      <td>18</td>
      <td>False</td>
      <td>F</td>
      <td>S</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0019_01</td>
      <td>Europa</td>
      <td>True</td>
      <td>C/0/S</td>
      <td>55 Cancri e</td>
      <td>31.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>19</td>
      <td>False</td>
      <td>C</td>
      <td>S</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0021_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>C/1/S</td>
      <td>TRAPPIST-1e</td>
      <td>38.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>6652.0</td>
      <td>0.0</td>
      <td>181.0</td>
      <td>585.0</td>
      <td>21</td>
      <td>False</td>
      <td>C</td>
      <td>S</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0023_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/5/S</td>
      <td>TRAPPIST-1e</td>
      <td>20.0</td>
      <td>False</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>635.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>23</td>
      <td>False</td>
      <td>F</td>
      <td>S</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



All looks good.

---

Now we are going to do some feature selection.


```python
df_train.dtypes
```




    PassengerId      object
    HomePlanet       object
    CryoSleep          bool
    Cabin            object
    Destination      object
    Age             float64
    VIP                bool
    RoomService     float64
    FoodCourt       float64
    ShoppingMall    float64
    Spa             float64
    VRDeck          float64
    Transported        bool
    Adults             bool
    Grouped            bool
    Deck             object
    Side             object
    Has_expenses       bool
    Is_Embryo          bool
    dtype: object




```python
features = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
            'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 
            'Grouped', 'Deck', 'Has_expenses', 'Side', 'Is_Embryo']
```

These are the features that I decided to use for model training and testing. I don't know if these are the best ones. So you can try different ones, and could even get a better result than mine!

Now we will assign the data in the training set to **feature** and **target** variables, and do a train-test-split split for evaluation.


```python
X = pd.get_dummies(df_train[features])
y = df_train['Transported']
```


```python
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)
```

Let's fit and score:


```python
model = LogisticRegression(max_iter=10000)
model.fit(X_train,y_train)
model.score(X_test,y_test)
```




    0.8003679852805887



Not bad.

Since we actually have to predict the **test** set that Kaggle has provided, we want to use all of the **train** data to train the model. The more data the model gets to learn from, the better the prediction.


```python
model2 = LogisticRegression(max_iter=10000)
model2.fit(X,y)
model2.score(X,y)
```




    0.792016565052341



Let's predict our test set now and save it:


```python
y_pred_log2 = model2.predict(pd.get_dummies(df_test[features]))
```

---

Now I'll use the only other classification model I knew at the time, **K-Neighbors Classifier**.


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```

And use **GridSearchCV** to get the optimal K value (code commented out as it takes time to run):


```python
# knn = KNeighborsClassifier()
# param_grid = {'n_neighbors':np.arange(2,15)}
# knn_gscv = GridSearchCV(knn, param_grid, cv=5)
# knn_gscv.fit(X,y)
# knn_gscv.best_params_
```


```python
knn2 = KNeighborsClassifier(n_neighbors=14)
knn2.fit(X,y)
knn2.score(X,y)
```




    0.8149085471068676



And save:


```python
y_pred_knn = knn2.predict(pd.get_dummies(df_test[features]))
```

---

Now, I did see that the model that seemed to perform great on this data is **Gradient Boosting Classifier**. So I looked it up and just used it with default hyperparameters:


```python
from sklearn.ensemble import GradientBoostingClassifier
gbr = GradientBoostingClassifier(random_state = 1)
  
# Fit to training set
gbr.fit(X, y)
gbr.score(X,y)
```




    0.8130679857356494



Seems slightly worse than our K-Neighbors Classifier. But still, we'll keep its predictions as well.


```python
pred_y_gbr = gbr.predict(pd.get_dummies((df_test[features])))
```

---

Since Gradient Boosting was performing well, and I had also stumbled upon **Extreme Gradient Boosting**, it only seems logical to try that out as well (maybe we'll get **extremely** good results):


```python
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X,y)
xgb.score(X,y)
```




    0.8887610721269987




```python
y_pred_xgb = xgb.predict(pd.get_dummies((df_test[features])))
```

---

The last thing I want to do is tune the Gradient Boost further using GSCV (my 4gb laptop dies when running this ok, so yes I will comment it out again):


```python
# gbc = GradientBoostingClassifier()
# parameters = {
#     "n_estimators":[5,50,100],
#     "max_depth":[1,3,5],
#    "learning_rate":[0.01,0.1,1]
# }

# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV

# cv = RandomizedSearchCV(gbc, parameters, n_iter=27, scoring='accuracy', n_jobs=-1, cv=5, random_state=1)
# cv.fit(X,y)
# cv.best_params_
```


```python
gbc1 = GradientBoostingClassifier(n_estimators=50,max_depth=5,learning_rate=0.1) #best params from gscv

gbc1.fit(X,y)
gbc1.score(X,y)
```




    0.831013459105027




```python
pred_y_gbr2 = gbc1.predict(pd.get_dummies((df_test[features])))
```

And so, we are done!

Time for submission.

# <p align="center"> Results </p>


```python
# Logist_out2 = pd.DataFrame({'PassengerId':df_test.PassengerId, 'Transported': y_pred_log2})
# Logist_out2.to_csv('submission.csv',index=False)
```

Logistic Regression competition Score = **0.79448**


```python
# knn_out = pd.DataFrame({'PassengerId':df_test.PassengerId, 'Transported': y_pred_knn})
# knn_out.to_csv('submission.csv',index=False)
```

KNN competition score = **0.79261**


```python
# xgb_out = pd.DataFrame({'PassengerId':df_test.PassengerId, 'Transported':y_pred_xgb.astype('bool')})
# xgb_out.to_csv('submission.csv',index=False)
```

XGB competition score = **0.79307**


```python
# gbr_out = pd.DataFrame({'PassengerId':df_test.PassengerId, 'Transported': pred_y_gbr})
# gbr_out.to_csv('submission.csv',index=False)
```

Gradient Boost competition score = **0.80056**


```python
gbc_out = pd.DataFrame({'PassengerId':df_test.PassengerId, 'Transported':pred_y_gbr2})
gbc_out.to_csv('submission.csv',index=False)
```

Tuned Gradient Boost competition score = **0.80476**

And so, we have a winner.

![GBC CM](https://user-images.githubusercontent.com/123200960/222893877-04c2cdf7-335c-4959-a7f0-39235d8bc657.png)

#
