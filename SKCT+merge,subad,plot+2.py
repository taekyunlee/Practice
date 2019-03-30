
# coding: utf-8

# In[3]:

import sklearn
import mglearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x, y = mglearn.datasets.make_wave(n_samples = 60) 
x_train, x_test , y_train , y_test = train_test_split(x,y , random_state = 42)

lr = LinearRegression().fit(x_train , y_train)


# In[4]:

lr.coef_


# In[5]:

lr.intercept_


# In[9]:

print("훈련세트 점수 {:.2f} ".format(lr.score(x_train, y_train)))


# In[10]:

from pandas import Series, DataFrame


# In[11]:

import pandas as pd


# In[12]:

import numpy as np


# In[14]:

df1 = pd.DataFrame({'key' :['b','b','a','c','a','a','b'], 'data1' : range(7)})
df2 = pd.DataFrame({'key':['a','b','d'], 'data2' : range(3)})


# In[15]:

df1


# In[16]:

df2


# In[17]:

pd.merge(df1,df2)


# In[18]:

pd.merge(df1, df2 , on = 'key')


# In[19]:

pd.merge(df1,df2, how = 'outer')


# In[21]:

df1


# In[22]:

df2


# In[23]:

pd.merge(df1,df2, how = 'inner')


# In[24]:

pd.merge(df1,df2)


# In[32]:

left1 = pd.DataFrame({'key' :['a','b','a','a','b','c'], 'value' : range(6)})
right1 = pd.DataFrame({'group_val' : [3.5 , 7]}, index = ['a','b'])


# In[35]:

left1


# In[36]:

right1


# In[37]:

pd.merge(left1 , right1 , left_on = 'key', right_index = True)


# In[38]:

s1 = Series([0,1], index = ['a','b'])
s2 = Series([2,3,4], index = ['c','d','e'])
s3 = Series([5,6], index = ['f','g'])


# In[40]:

pd.concat([s1,s2,s3] , axis = 1)


# In[41]:

s4 = pd.concat([s1*5, s3])


# In[42]:

s4


# In[43]:

pd.concat([s1,s4], axis =1)


# In[44]:

pd.concat([s1,s4], axis =1 , join = 'inner')


# In[45]:

s1


# In[47]:

s4


# In[48]:

from pandas import Series , DataFrame


# In[49]:

result = pd.concat([s1,s3,s3], keys = ['one','two','three'])


# In[50]:

result


# In[51]:

result.unstack()


# In[52]:

df1 = pd.DataFrame(np.arange(6).reshape(3,2), index = ['a','b','c'], columns = ['one','two'])
df2 = pd.DataFrame(5+ np.arange(4).reshape(2,2) , index = ['a','c'], columns = ['three','four'])


# In[53]:

df1


# In[54]:

df2


# In[55]:

pd.concat([df1, df2 ] , axis =1 , keys = ['level1','level2'])


# In[56]:

pd.concat([df1, df2 ] ,  keys = ['level1','level2'])


# In[57]:

df1 = pd.DataFrame(np.random.randn(3,4),  columns = ['a','b','c','d'])
df2 = pd.DataFrame(np.random.randn(2,3) , columns = ['b','d','a'])


# In[58]:

df1


# In[59]:

df2


# In[60]:

pd.concat([df1,df2], ignore_index = True)


# In[62]:

a = Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan], index = ['f','e','d','c','b','a'])
b = Series(np.arange(len(a), dtype = np.float64), index = ['f','e','d','c','b','a'])


# In[63]:

a


# In[64]:

b


# In[66]:

b[:-2].combine_first(a[2:])


# In[67]:

df1 = pd.DataFrame({'a': [1., np.nan, 5., np.nan],
                    'b' : [np.nan, 2. , np.nan , 6.],
                    'c' : range(2,18,4)})


# In[68]:

df2 = pd.DataFrame({'a': [5.,4.,np.nan, 3., 7.], 
                    'b': [np.nan, 3., 4., 6., 8.]})


# In[69]:

df1


# In[70]:

df2


# In[71]:

df1.combine_first(df2)


# In[73]:

str.lower('A')


# In[74]:

'A'.lower()


# In[75]:

a = {'a' : 1}


# In[76]:

a


# In[78]:

a['a']


# In[79]:

import scipy


# In[80]:

scipy.nan


# In[81]:

np.nan


# In[87]:

data = pd.DataFrame(np.arange(12).reshape((3,4)), index = ['Ohio','colorado','new york'], 
                   columns = ['one','two','three','four'])


# In[88]:

data


# In[89]:

data.index.map(str.lower)


# In[91]:

data.index = data.index.map(str.lower)


# In[92]:

data


# In[93]:

data


# In[95]:

data.rename(index = str.title, columns = str.upper)


# In[96]:

data


# In[97]:

ages = [20, 22, 25, 27, 21, 23, 37, 61, 45, 41, 32]


# In[98]:

bins = [18, 25, 35, 60 ,100]


# In[99]:

cats = pd.cut(ages, bins)


# In[100]:

cats 


# In[101]:

pd.value_counts(cats)


# In[104]:

scipy.random.seed(12345)

data = DataFrame(scipy.random.randn(1000,4))


# In[105]:

data.describe()


# In[106]:

data


# In[108]:

col = data[3]


# In[109]:

col[scipy.absolute(col) >3 ]


# In[110]:

data


# In[113]:

data[ (scipy.absolute(data) >3).any(1) ]


# In[120]:

import scipy as s 
data[(scipy.absolute(data) >3)] = s.sign(data)*3


# In[121]:

data


# In[122]:

df = DataFrame(np.arange(5*4).reshape(5,4))


# In[123]:

sampler = np.random.permutation(5)


# In[124]:

sampler


# In[126]:

s.rand(100).shape


# In[127]:

import scipy as sp


# In[129]:

sp.absolute


# In[158]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


# In[159]:

fig  = plt.figure()


# In[160]:

ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)


# In[163]:

plt.plot(sp.randn(50).cumsum(), 'k--')


# In[171]:

sp.randn(50).cumsum()


# In[175]:

_ = ax1.hist(sp.randn(100), bins = 20, color = 'k', alpha = 0.3)


# In[177]:

get_ipython().magic('matplotlib inline')
ax1


# In[179]:

fig, axes = plt.subplots(2,2)


# In[181]:

fig ,axes = plt.subplots(2,2, sharex = True , sharey = True) 
for i in range(2) :
     for j in range(2) : 
            axes[i,j].hist(sp.randn(500), bins = 50 , color ='k', alpha = 0.5)
plt.subplots_adjust(wspace=  0 , hspace = 0 )


# In[184]:

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(sp.randn(1000).cumsum())


# In[190]:

ax.set_xticks([0,250, 500, 750, 1000])
#ax.set_xticklabels(['one','two','three','four','five'], rotation = 30 , fontsize = 'small')


# In[191]:

fig


# In[193]:

df = DataFrame(np.random.randn(10,4).cumsum(0), columns = ['A','B','C','D'], index = np.arange(0, 100 ,10))


# In[194]:

df


# In[195]:

df.plot()


# In[197]:

fig, axes = plt.subplots(2,1)

data = Series(np.random.rand(16), index = list('abcdefghijklmnop'))
data.plot(kind = 'bar' , ax = axes[0], color = 'k' , alpha = 0.7)
data.plot(kind = 'barh' , ax = axes[1], color = 'k', alpha = 0.7)


# In[ ]:



