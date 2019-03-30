
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
y_train = data['log_price']
x_test = test.drop(['id','date'], axis = 1)


# In[17]:

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error

cross_val_score(LinearRegression() , x_train , y_train, cv = 5)    # train 가지고 cv돌려서 rmse구하기?


# In[52]:

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5)
cross_val_score(LinearRegression(), x_train , y_train , cv =kfold)


# In[53]:

scores = cross_val_score(LinearRegression(), x_train , y_train , cv =kfold, scoring = 'mean_squared_error')


# In[54]:

sp.sqrt(-scores).mean()


# In[59]:

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor


kfold = KFold(n_splits=5)
scores = cross_val_score(RandomForestRegressor(n_estimators = 100), x_train , y_train , cv =kfold, scoring = 'mean_squared_error')
scores
# sp.sqrt(-scores).mean()  # y 변환 안해줌 


# In[57]:

x_train


# In[58]:

y_train


# In[65]:

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
scores = cross_val_score(RandomForestRegressor(n_estimators = 100), x_train , y_train , cv =kfold, scoring = 'neg_mean_squared_error')


# In[63]:

sp.sqrt(-scores).mean()


# In[64]:

x_train = data.drop(['id','date','price','log_price','per_price'], axis = 1 )
y_train = data['log_price']
x_test = test.drop(['id','date'], axis = 1)

ran = RandomForestRegressor(n_estimators = 100).fit(x_train.values, y_train.values)
ran.predict(x_test.values)
pred = sp.exp(ran.predict(x_test.values)) - 1 
sample['price'] = pred
sample.to_csv("C:/Users/SS/Desktop/2019-2nd-ml-month-with-kakr/sampleranfo.csv", index = False)


# In[ ]:

-2.11361358e+10, -1.88994450e+10, -1.47820929e+10, -1.60494754e+10,
       -1.57061159e+10])


# In[66]:

scores


# In[ ]:



