
# coding: utf-8

# In[1]:

import pandas as pd
from pandas import Series, DataFrame
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
get_ipython().magic('matplotlib inline')


# In[50]:

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

param_grid = {'C':[0.001,0.01,0.1,10,100],
             'gamma' : [0.001, 0.01,0.1, 1, 10, 100]}

grid_search = GridSearchCV(SVC(), param_grid , cv=5)

iris = load_iris()


# In[65]:

x_train, x_test , y_train, y_test = train_test_split(iris.data, iris.target)
grid_search.fit(x_train, y_train) 


# In[66]:

grid_search.score(x_test, y_test)


# In[67]:

grid_search.best_params_


# In[68]:

grid_search.best_estimator_


# In[69]:

grid_search.best_score_


# In[70]:

import pandas as pd
from pandas import Series, DataFrame
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
get_ipython().magic('matplotlib inline')

data = pd.read_csv("C:/Users/SS/Desktop/2019-2nd-ml-month-with-kakr/train.csv")
test = pd.read_csv("C:/Users/SS/Desktop/2019-2nd-ml-month-with-kakr/test.csv")
sample = pd.read_csv("C:/Users/SS/Desktop/2019-2nd-ml-month-with-kakr/sample_submission.csv")

data['log_price'] = sp.log1p(data['price'])

data = data[data.sqft_living != 13540]
data= data[data.id != 2302]
data= data[data.id != 4123]
data= data[data.id != 7173]
data= data[data.id != 2775]

for i in ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement'] : 
      
        data[i] = sp.log1p(data[i])
        test[i] = sp.log1p(test[i])
        
data['date'] = data['date'].apply(lambda x : x[0:8])
test['date'] = test['date'].apply(lambda x : x[0:8])

data['yr_renovated'] = data['yr_renovated'].apply(lambda x : sp.nan if x == 0 else x )
data['yr_renovated'] = data['yr_renovated'].fillna(data['yr_built'])

test['yr_renovated'] = test['yr_renovated'].apply(lambda x : sp.nan if x == 0 else x )
test['yr_renovated'] = test['yr_renovated'].fillna(test['yr_built'])

for df in [data, test] : 
    
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['sqft_ratio'] = df['sqft_living'] / df['sqft_lot']
    df['sqft_total_size'] = df['sqft_above'] + df['sqft_basement']
    df['sqft_ratio15'] = df['sqft_living15'] / df['sqft_lot15'] 
    df['is_renovated'] = df['yr_renovated'] - df['yr_built']
    df['is_renovated'] = df['is_renovated'].apply(lambda x: 0 if x == 0 else 1)
    df['date'] = df['date'].astype('int')
    
data['per_price'] = data['price']/ data['sqft_total_size']
zipcode_price = data.groupby(['zipcode'])['per_price'].agg({'mean','var'}).reset_index()

data = pd.merge(data, zipcode_price,how='left', on='zipcode')
test = pd.merge(test, zipcode_price,how='left', on='zipcode')

for df in [data, test]:
    
    df['mean'] = df['mean'] * df['sqft_total_size']
    df['var'] = df['var'] * df['sqft_total_size']
    
x_train = data.drop(['id','date','price','log_price','per_price'], axis = 1 )
y_train = data['price']
x_test = test.drop(['id','date'], axis = 1)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

kfold = KFold(n_splits=5)
#scores = cross_val_score(RandomForestRegressor(n_estimators = 100), x_train , y_train , cv =kfold, scoring = 'neg_mean_squared_error')


# In[76]:

from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators' :[50, 100]}
grid_search = GridSearchCV(RandomForestRegressor() ,  param_grid  , cv = kfold, scoring= 'mean_squared_error')

grid_search.fit(x_train , y_train)


# In[77]:

grid_search.best_params_


# In[78]:

grid_search.best_estimator_


# In[82]:

grid_search.best_score_


# In[84]:

sp.sqrt(-(grid_search.best_score_))


# In[ ]:



