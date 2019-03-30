
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


# In[7]:

data= pd.read_csv("C:/Users/SS/Desktop/WA_Sales_Products_2012-14.csv", engine ='python', encoding ='cp949')


# In[16]:

from scipy import stats
stats.ttest_1samp(data.Revenue, 40000)


# In[11]:

data.Revenue.mean()


# In[20]:

data.describe()


# In[26]:

data.columns
Outdoor = data[data['Retailer type'] == "Outdoors Shop"] #Outdoor shop에서 물건을 산 고객
Department = data[data['Retailer type'] == "Department Store"] #Department store에서 물건을 산 고객


# In[28]:

stats.levene(Outdoor.Revenue, Department.Revenue) # 등분산이 아님
stats.ttest_ind(Outdoor.Revenue, Department.Revenue, equal_var=False)
# outdoor와 department 매출액의 평균 차이가 있다고 할 수 있다.


# In[87]:

data = pd.DataFrame({'pass' : [0,1,1,1,0], 'score' : [380, 660, 800, 640 , 520], 'gpa' : [3.61,3.67,4.0,3.19 ,2.93]})


# In[99]:

import statsmodels.api as sm

data = pd.DataFrame({'pass' : [0,1,1,1,0], 'score' : [380, 660, 800, 640 , 520], 'gpa' : [3.61 ,3.67 ,4.0 , 3.19 ,2.93]})

x = data[['score','gpa']]
y = data[['pass']]


logit = sm.Logit(y , x)
logit.fit()


# In[106]:

#from sklearn.model_selection import train_test_split
#x_train , x_test, y_train , y_test = train_test_split(x,y, test_size = 0.3, random_state = 0 )

import statsmodels.api as sm

X = pd.DataFrame({'x1': [3.61 ,3.67 , 4.0 , 3.19 ,2.93], 'x2': [380, 660, 800, 640 , 520]})
y = pd.DataFrame({'y': [0,1,1,1,0]})

logit = sm.Logit(y, X)
result = logit.fit()
print(result.summary2())


# In[ ]:




# In[108]:

import statsmodels.api as sm

X = pd.DataFrame({'x1': [2, 4, 3, 3.5, 2, 5.5, 1], 'x2': [1, 1.5, 1, 0.5, 0.5, 1, 1]})
y = pd.DataFrame({'y': [0, 1, 0, 1, 0, 1, 0]})

logit = sm.Logit(y, X)
result = logit.fit()
print(result.summary2())


# In[ ]:




# In[ ]:



