
# coding: utf-8

# In[606]:

import pandas as pd
from pandas import Series, DataFrame
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
get_ipython().magic('matplotlib inline')

data = pd.read_csv("C:/Users/SS/Desktop/2019-2nd-ml-month-with-kakr/train.csv")
test = pd.read_csv("C:/Users/SS/Desktop/2019-2nd-ml-month-with-kakr/test.csv")
sample     = pd.read_csv("C:/Users/SS/Desktop/2019-2nd-ml-month-with-kakr/sample_submission.csv")


# In[525]:

data


# In[526]:

data.info()
data.columns


# In[527]:

data.describe()


# In[528]:

data.nunique()


# In[529]:

col = data.loc[ :, 'price' : 'sqft_lot15'].columns
col

fig , axes = plt.subplots(19, figsize = (10,10))
    
for i,j in enumerate(col) : 
              
        sns.distplot(data[j], ax= axes[i])
        axes[i].set_xlabel(col[i])
        plt.subplots_adjust(top = 10)
       
    


# In[530]:

sns.pairplot(data)


# In[531]:

data.isnull().sum()


# In[532]:

plt.figure(figsize = (20,20))
fig, axes = plt.subplots(7,3)

k = 0

for i in range(7) : 
    for j in range(3) :
    
         sns.boxplot(data[col[k]], ax = axes[i,j], orient = "v" )
         axes[i,j].set_ylabel(col[k], fontsize = 30)
         k = k+1
         plt.subplots_adjust(top =20, right =5 )
         if k == 19 :
              break


# In[533]:

data.corr()[data.corr() > 0.5]


# In[534]:

plt.figure(figsize = (15,10))
sns.heatmap(data.corr(), annot =True)


# In[535]:

data.columns


# In[536]:

sns.boxplot(data.bedrooms, data.price, data = data)


# In[537]:

pd.crosstab(data.bedrooms, data.grade)


# In[538]:

data


# In[539]:

data['price'].describe()


# In[540]:

plt.figure(figsize = (8,7))
sns.distplot(data.price)


# In[541]:

data.price.skew() # 왜도
#data.price.kurt() # 첨도


# In[542]:

data['log_price'] = sp.log1p(data['price'])


# In[543]:

sns.distplot(data.log_price)


# In[544]:

plt.hist(data.log_price) #빈도수


# In[545]:

a = (data.corr()[data.corr() > 0.5].sum() > 1)


# In[546]:

plt.figure(figsize = (10,7))
sns.heatmap(data[a[ a== True].index].corr(), annot = True )


# In[547]:

b = data.corr()[data.corr() > 0.4]


# In[548]:

c = b.reset_index()[(b.reset_index() > 0.4) & (b.reset_index() < 1)].isnull().all()


# In[549]:

plt.figure(figsize = (10,7))
sns.heatmap(data[c[ c == False].index.drop('index')].corr(), annot = True )


# In[550]:

data.corr()


# In[551]:

plt.figure(figsize = (10,10))
sns.boxplot(data.grade, data.log_price, data= data)


# In[552]:

plt.figure(figsize = (10,10))
sns.regplot('sqft_living', 'log_price' , data = data)


# In[553]:

plt.figure(figsize = (10,10))
sns.regplot('sqft_living15', 'log_price' , data = data)


# In[554]:

plt.figure(figsize = (10,10))
sns.regplot('sqft_above', 'log_price' , data = data)


# In[555]:

plt.figure(figsize = (13,7))
sns.boxplot(data.bathrooms, data.log_price, data= data)


# In[556]:

plt.figure(figsize = (13,7))
sns.boxplot(data.bedrooms, data.log_price, data= data)


# In[557]:

data.isnull().sum().plot( 'bar')


# In[558]:

plt.figure(figsize = (13,7))
data.nunique().plot('bar')


# In[559]:

data.waterfront.value_counts()
data.view.value_counts()


# In[560]:

plt.figure(figsize = (5,5))
sns.regplot('sqft_living', 'log_price' , data = data)


# In[561]:

data[data.sqft_living > 13000]


# In[562]:

data = data[data.sqft_living != 13540]


# In[563]:

data[data.sqft_living > 13000]


# In[564]:

plt.figure(figsize = (10,10))
sns.boxplot(data.grade, data.log_price, data= data)


# In[565]:

data[(data.grade ==3) & (data.log_price > 12)]


# In[566]:

data[(data.grade == 8 ) & (data.log_price > 14.8)]


# In[567]:

data[(data.grade == 11 ) & (data.log_price > 15.5)]


# In[568]:

data= data[data.id != 2302]
data= data[data.id != 4123]
data= data[data.id != 7173]
data= data[data.id != 2775]


# In[569]:

sp.log1p(10)


# In[570]:

sp.log(11)


# In[572]:

#train / test 


# In[573]:

for i in ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement'] : 
      
        data[i] = sp.log1p(data[i])
        test[i] = sp.log1p(test[i])


# In[574]:

data['date'] = data['date'].apply(lambda x : x[0:8])
test['date'] = test['date'].apply(lambda x : x[0:8])


# In[575]:

data.yr_renovated.value_counts()


# In[576]:

data['yr_renovated'] = data['yr_renovated'].apply(lambda x : sp.nan if x == 0 else x )
data['yr_renovated'] = data['yr_renovated'].fillna(data['yr_built'])

test['yr_renovated'] = test['yr_renovated'].apply(lambda x : sp.nan if x == 0 else x )
test['yr_renovated'] = test['yr_renovated'].fillna(test['yr_built'])


# In[577]:

data.head()


# In[578]:

for df in [data, test] : 
    
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['sqft_ratio'] = df['sqft_living'] / df['sqft_lot']
    df['sqft_total_size'] = df['sqft_above'] + df['sqft_basement']
    df['sqft_ratio15'] = df['sqft_living15'] / df['sqft_lot15'] 
    df['is_renovated'] = df['yr_renovated'] - df['yr_built']
    df['is_renovated'] = df['is_renovated'].apply(lambda x: 0 if x == 0 else 1)
    df['date'] = df['date'].astype('int')


# In[579]:

data[['sqft_living','sqft_total_size']]


# In[580]:

data['per_price'] = data['price']/ data['sqft_total_size']  # price per sqft_total size


# In[581]:

zipcode_price = data.groupby(['zipcode'])['per_price'].agg({'mean','var'}).reset_index() # 구역별 평당 가격


# In[582]:

zipcode_price


# In[583]:

data = pd.merge(data, zipcode_price,how='left', on='zipcode')
test = pd.merge(test, zipcode_price,how='left', on='zipcode')


# In[584]:

data.shape


# In[585]:

data.columns


# In[586]:

for df in [data, test]:
    
    df['mean'] = df['mean'] * df['sqft_total_size']
    df['var'] = df['var'] * df['sqft_total_size']


# In[587]:

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# In[588]:

col = data.columns.drop(['id','price', 'log_price','per_price'])


# In[589]:

model = sm.OLS(data['log_price'].values, data[col])
result = model.fit()
print(result.summary())


# In[590]:

3.786e-01


# In[591]:

data.columns


# In[592]:

test.columns


# In[594]:

x_train = data.drop(['id','date','price','log_price','per_price'], axis = 1 )
y_train = data['log_price']


# In[595]:

x_test = test.drop(['id','date'], axis = 1 )


# In[596]:

lin = LinearRegression().fit(x_train.values, y_train.values)
lin.predict(x_test.values)


# In[602]:

pred = sp.exp(lin.predict(x_test.values)) - 1


# In[605]:

pred


# In[608]:

sample['price'] = pred


# In[611]:

sample
sample.to_csv("C:/Users/SS/Desktop/2019-2nd-ml-month-with-kakr/sample.csv", index = False)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



