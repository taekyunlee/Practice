
# coding: utf-8

# In[29]:

import pandas as pd
from pandas import DataFrame, Series 
import scipy as sp
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')

train = pd.read_csv("C:/Users/SS/Desktop/titanic/train.csv")
test = pd.read_csv("C:/Users/SS/Desktop/titanic/test.csv")

train.Embarked.fillna('C', inplace = True)

survivers = train.Survived

train.drop(["Survived"],axis=1, inplace=True)

all_data = pd.concat([train,test], ignore_index=False)

## Assign all the null values to N
all_data.Cabin.fillna("N", inplace=True)
all_data.Cabin = [i[0] for i in all_data.Cabin] # all_data.Cabin = all_data.Cabin.apply(lambda x : x[0])

with_N = all_data[all_data.Cabin == "N"]

without_N = all_data[all_data.Cabin != "N"]

all_data.groupby('Cabin')['Fare'].mean().sort_values()


def cabin_estimator(i):
    
    if i<16:
        a = "G"
    elif i>=16 and i<27:
        a = "F"
    elif i>=27 and i<38:
        a = "T"
    elif i>=38 and i<47:
        a = "A"
    elif i>= 47 and i<53:
        a = "E"
    elif i>= 53 and i<54:
        a = "D"
    elif i>=54 and i<116:
        a = 'C'
    else:
        a = "B"
    return a

with_N['Cabin'] = with_N['Fare'].apply(lambda x :cabin_estimator(x))

all_data = pd.concat([with_N, without_N], axis = 0 )
all_data.sort_values(by = 'PassengerId', inplace=True)

train = all_data[ : 891]
test = all_data[891 : ]

train['Survived'] = survivers

missing_value = test[(test.Pclass ==3) & (test.Sex == 'male') & (test.Embarked == 'S')].Fare.mean()
test['Fare'].fillna(missing_value, inplace = True)


# In[3]:

train.shape


# In[4]:

test.shape


# In[5]:

train


# In[6]:

test


# In[30]:

train.info()


# In[19]:

#train = train.drop(['PassengerId','Name','Ticket'], axis =1 )
#test = test.drop(['PassengerId','Name','Ticket'], axis =1 )


# In[26]:

#col = [i for i in train.columns]
#col.remove('Survived')

#x_train = train[col]
#y_train = train['Survived']


# In[33]:

#from sklearn.ensemble import RandomForestClassifier
#ran = RandomForestClassifier().fit(x_train , y_train)


# In[31]:

train = train.drop(['PassengerId','Name','Ticket','Age'], axis =1 )
test = test.drop(['PassengerId','Name','Ticket','Age'], axis =1 )

integ = pd.concat([train,test])
integ = pd.get_dummies(integ)


# In[32]:

train = integ.reset_index()[ : 891]
test = integ.reset_index()[891 : ]

col = [i for i in train.columns]
col.remove('Survived')

x_train = train[col]
y_train = train['Survived']

x_train = x_train.drop(['index'], axis =1 )
test = test.drop(['index','Survived'], axis =1 )


# In[38]:

y_train = y_train.astype(int).astype(str)


# In[39]:

from sklearn.ensemble import RandomForestClassifier
ran = RandomForestClassifier().fit(x_train.values , y_train.values)


# In[43]:

ran.predict(test.values)


# In[44]:

sample = pd.read_csv("C:/Users/SS/Desktop/titanic/gender_submission.csv")


# In[45]:

sample['Survived'] = ran.predict(test.values)


# In[46]:

sample.to_csv("C:/Users/SS/Desktop/titanicc.csv", index= False)


# In[26]:

sample.info()


# In[27]:

train.columns


# In[28]:

train['Survived']


# In[68]:

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

param_grid = {'n_estimators' :[2,3,5,10,100,300]}
grid = GridSearchCV(GradientBoostingClassifier(), param_grid = param_grid, cv =5)
grid.fit(x_train , y_train)


# In[69]:

grid.best_score_


# In[70]:

grid.best_params_


# In[ ]:



