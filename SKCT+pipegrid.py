
# coding: utf-8

# In[16]:

import pandas as pd
from pandas import Series, DataFrame
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
get_ipython().magic('matplotlib inline')


# In[17]:

cancer = load_breast_cancer()
x_train , x_test , y_train, y_test = train_test_split(cancer.data, cancer.target)
pipe = Pipeline([('scaler', MinMaxScaler()), ("svm" , SVC())])
pipe.fit(x_train , y_train)
pipe.score(x_test, y_test)


# In[19]:

param_grid = {'svm__C' : [0.001, 0.01, 0.1, 1, 10, 100], 
              'svm__gamma' : [0.001, 0.01, 0.1, 1, 10 ,100]}

grid = GridSearchCV(pipe, param_grid = param_grid , cv =5 )
grid.fit(x_train , y_train)
print(grid.best_score_)
print(grid.score(x_test, y_test))


# In[20]:

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pipe = make_pipeline(StandardScaler(), PCA(n_components = 2), StandardScaler())


# In[24]:

pipe.fit(cancer.data)

components = pipe.named_steps
components['pca'].components_.shape


# In[31]:

from sklearn.linear_model import LogisticRegression
pipe = make_pipeline(StandardScaler(), LogisticRegression())

param_grid = {'logisticregression__C' : [0.01,0.1,1,10,100]}

x_train ,x_test , y_train , y_test = train_test_split(cancer.data ,cancer.target, random_state =4)
grid = GridSearchCV(pipe , param_grid= param_grid , cv =5 )
grid.fit(x_train , y_train)


# In[37]:

grid.best_estimator_.named_steps['logisticregression'].coef_.shape


# In[42]:

from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
boston = load_boston()
x_train , x_test, y_train , y_test = train_test_split(boston.data, boston.target, random_state = 0)

from sklearn.preprocessing import PolynomialFeatures
pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())

param_grid = {'polynomialfeatures__degree' : [1,2,3], 
             'ridge__alpha' :[0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(pipe, param_grid= param_grid, cv = 5 , n_jobs = -1)
grid.fit(x_train , y_train)


# In[43]:

grid.best_params_


# In[44]:

grid.score(x_test , y_test)


# In[ ]:



