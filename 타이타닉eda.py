
# coding: utf-8

# In[103]:

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


# In[8]:

train.head()


# In[35]:

test.head()


# In[33]:

train.info()
print(test.info())


# In[16]:

sns.countplot(train['Parch'])
sns.countplot(train.SibSp)


# In[32]:

print(train.isnull().sum())
test.isnull().sum()


# In[24]:

sns.boxplot(train.Age, orient = 'v')


# In[25]:

train.columns


# In[44]:

plt.figure(figsize= (15,10))
train.groupby(train.Age).count()['Survived'].plot(kind = 'bar')


# In[106]:

train['Age'] = train['Age'].apply(lambda x : 0 if 0<x<10 else x)
train['Age'] = train['Age'].apply(lambda x : 10 if 10<=x<20 else x)
train['Age'] = train['Age'].apply(lambda x : 20 if 20<=x<30 else x)
train['Age'] = train['Age'].apply(lambda x : 30 if 30<=x<40 else x)
train['Age'] = train['Age'].apply(lambda x : 40 if 40<=x<50 else x)
train['Age'] = train['Age'].apply(lambda x : 50 if 50<=x<60 else x)
train['Age'] = train['Age'].apply(lambda x : 60 if 60<=x<70 else x)
train['Age'] = train['Age'].apply(lambda x : 70 if 70<=x<80 else x)
train['Age'] = train['Age'].apply(lambda x : 80 if 80<=x<90 else x)


# In[117]:

train[train['Age'].isnull()]


# In[118]:

pd.crosstab(train.Age, train.Survived)


# In[120]:

plt.scatter(train.Fare , train.Survived)


# In[123]:

sns.distplot(train.Fare)


# In[124]:

train.corr()


# In[128]:

sns.pairplot(train[['Survived','Pclass','SibSp','Parch','Fare']]) #nan 제거


# In[129]:

train


# In[133]:

train.groupby(train.Age).count()


# In[135]:

pd.crosstab(train.Age, train.Survived).plot()


# In[139]:

plt.figure(figsize = (7,7))
sns.heatmap(pd.crosstab(train.Age, train.Survived), annot= True)


# In[141]:

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


# In[151]:

sns.distplot(sp.random.normal(size=(100)))


# In[153]:

sns.distplot(sp.random.exponential(size=100))


# In[155]:

data = pd.DataFrame({'number' : [1,2,3,4,5,6,7,8,9,10] , 'ob' : ['a','b','c','d','e','f','g','f','i','g']})


# In[169]:

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data[['number']])


# In[197]:

new = pd.DataFrame(scaler.transform(data[['number']])).rename(columns = {0 : 'number1'})


# In[198]:

new = new.reset_index().drop('index', axis =1 )
new


# In[204]:

pd.concat([new, data] , axis = 1).drop('number', axis =1 ).rename(columns = {'number1' : 'number'})


# In[218]:

data1 = pd.get_dummies(data)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler


# In[219]:

scaler = RobustScaler()
scaler.fit(data1)
scaler.transform(data1).shape
pd.DataFrame(scaler.transform(data1))


# In[212]:

data1


# In[214]:

pd.DataFrame(scaler.transform(data1))


# In[221]:

train.isnull().sum().sum()


# In[224]:

(train.isnull().sum() / train.isnull().sum().sum() ) *100


# In[239]:

a = pd.DataFrame(train.isnull().sum())


# In[227]:

len(train)


# In[228]:

train.isnull().sum().sum() 


# In[232]:

b = pd.DataFrame((train.isnull().sum() / train.isnull().sum().sum() ) *100)


# In[236]:

b = b.rename(columns = {0 : 'percent'})


# In[240]:

pd.concat([a,b] , axis =1 )


# In[242]:

train[train.Embarked.isnull()]


# In[243]:

pd.crosstab(train.Embarked, train.Pclass)


# In[252]:

plt.figure(figsize = (10, 10))
sns.boxplot(train.Embarked, train.Fare , hue = train.Pclass)


# In[254]:

plt.figure(figsize = (10, 10))
sns.boxplot(test.Embarked, test.Fare , hue = test.Parch)


# In[250]:

plt.figure(figsize = (10, 10))
sns.boxplot(test.Embarked, test.Fare ) 


# In[258]:

train.Embarked.fillna('C', inplace = True)


# In[259]:

train.isnull().sum()


# In[260]:

train.Cabin


# In[263]:

survivers = train.Survived

train.drop(["Survived"],axis=1, inplace=True)

all_data = pd.concat([train,test], ignore_index=True)

## Assign all the null values to N
all_data.Cabin.fillna("N", inplace=True)


# In[323]:

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


# In[320]:

train.isnull().sum()


# In[331]:

test.isnull().sum()


# In[326]:

test[test.Fare.isnull()]


# In[332]:

sns.barplot(train.Sex, train.Survived)


# In[333]:

pd.crosstab(train.Sex, train.Survived)


# In[337]:

sns.countplot(train.Sex, hue = train.Survived)


# In[338]:

sns.barplot(train.Pclass, train.Survived)


# In[345]:

sns.distplot(train.Pclass[train.Survived == 0] , label = 'not')
sns.distplot(train.Pclass[train.Survived == 1] , label = 'sur')
plt.legend()


# In[350]:

sns.kdeplot(train[train.Survived ==0].Fare, label = 'no')
sns.kdeplot(train[train.Survived ==1].Fare, label = 'yes')


# In[352]:

sns.kdeplot(train[train.Survived ==0].Age, label = 'no')
sns.kdeplot(train[train.Survived ==1].Age, label = 'yes')


# In[357]:

pal = {1:"seagreen", 0:"gray"}
sns.FacetGrid(train, col="Sex", row="Embarked", hue = "Survived", palette = pal)


# In[358]:

train.corr()


# In[ ]:




# In[ ]:



