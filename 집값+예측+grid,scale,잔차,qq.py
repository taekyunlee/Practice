
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
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA 
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
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

integ = pd.concat([data, test])
integ['view'] = integ['view'].astype(str)

integ = pd.get_dummies(integ)

train = integ[: 15030]
test = integ[15030 : ]

train = train.drop(['id','price','zipcode','per_price'], axis =1 )
test = test.drop(['id','zipcode','price','log_price','per_price'], axis =1)

col = [i for i in train.columns]
col.remove('log_price')

x_train = train[col]
y_train = train['log_price']


# In[119]:

param_grid = {'lasso__alpha' : [0.001, 0.01,0.1,1,10,100, 1000]}
pipe = make_pipeline(MinMaxScaler(), PCA(n_components=30), Lasso()) 

grid = GridSearchCV(pipe , param_grid = param_grid , cv = 5, scoring='mean_squared_error')
grid.fit(x_train , y_train)


# In[144]:

scaler = MinMaxScaler().fit(x_train)
scaler.transform(x_train)
scaler.transform(test)


# In[155]:

lasso = Lasso(0.001).fit(scaler.transform(x_train), y_train).predict(scaler.transform(test))


# In[157]:

lasso


# In[159]:

sp.exp(lasso) - 1


# In[163]:

sample['price'] = sp.exp(lasso) - 1 
sample.to_csv("C:/Users/SS/Desktop/lasso.csv", index = False)


# In[ ]:

scaler = MinMaxScaler().fit(x_train)
scaler.transform(x_train)
scaler.transform(test)


# In[170]:

pca = PCA(n_components=28).fit(scaler.transform(x_train))
pca.transform(scaler.transform(x_train))
pca.transform(scaler.transform(test))


# In[171]:

lasso = Lasso(0.001).fit(pca.transform(scaler.transform(x_train)), y_train).predict(pca.transform(scaler.transform(test)))
sp.exp(lasso) - 1 
sample['price'] = sp.exp(lasso) - 1 
sample.to_csv("C:/Users/SS/Desktop/lasso.csv", index = False)


# In[173]:

from sklearn.ensemble import GradientBoostingRegressor
gra = GradientBoostingRegressor().fit(x_train.values ,y_train.values).predict(test.values)
sample['price'] = sp.exp(gra) - 1 
sample.to_csv("C:/Users/SS/Desktop/gra.csv", index = False)


# In[7]:

plt.figure(figsize = (10,7) )
plt.scatter(data.sqft_living , data.price)


# In[10]:

import statsmodels.api as sm
result = sm.OLS(data.price, data.sqft_living).fit()
result.summary()


# In[17]:

plt.scatter(data.sqft_living, result.resid)


# In[29]:

from sklearn.linear_model import LinearRegression

lin = LinearRegression().fit(data[['sqft_living']].values, data[['price']].values)


# In[35]:

print(lin.coef_)
print(lin.intercept_)


# In[37]:

y = -3448927.83050369 + 528182.09576692*data[['sqft_living']].values


# In[39]:

y.shape


# In[53]:

plt.figure(figsize = (10,10))
plt.scatter(data[['sqft_living']].values,  data[['price']].values - y ) # 등분산성, 잔차의 정규성 위배 -> 단순 선형회귀는 적절치않음.


# In[49]:

data[['price']].values - y


# In[52]:

sns.regplot(data.sqft_living, data.price)


# In[56]:

sp.stats.probplot(result.resid, plot =plt) # Q-Q Plot


# In[ ]:



