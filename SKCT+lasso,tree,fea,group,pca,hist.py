
# coding: utf-8

# In[25]:

import sklearn
import pandas as pd
import mglearn
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

x,y = mglearn.datasets.load_extended_boston()

x_train, x_test , y_train , y_test = train_test_split(x,y , random_state = 0)
ridge = Ridge().fit(x_train, y_train)

ridge.score(x_train, y_train)

ridge10 = Ridge(10).fit(x_train , y_train)
ridge01 = Ridge(0.1).fit(x_train , y_train)


# In[23]:

plt.plot(ridge10.coef_, '^', label = '10')
plt.plot(ridge.coef_, 's', label = '1')
plt.plot(ridge01.coef_, 'v', label = '0.1')

plt.legend()
plt.hlines(0,2,len(lr.coef_))


# In[15]:

from sklearn.linear_model import Lasso


# In[24]:

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

x,y = mglearn.datasets.make_forge()

fig, axes = plt.subplots(1,2, figsize = (10,3))

for model, ax in zip([LinearSVC(), LogisticRegression()], axes) :
    clf = model.fit(x,y)
    mglearn.plots.plot_2d_separator(clf, x, fill = False , eps= 0.5, ax= ax , alpha = 0.7)
    mglearn.discrete_scatter(x[:,0], x[:,1], y, ax =ax )
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("특성 0")
    ax.set_ylabel("특성 1")
axes[0].legend()


# In[27]:

sp.linspace(-15, 15).shape


# In[29]:

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
x_train , x_test, y_train , y_test = train_test_split(cancer.data ,cancer.target, stratify = cancer.target, random_state = 42)

tree = DecisionTreeClassifier(random_state = 0)
tree.fit(x_train, y_train)


# In[34]:

tree.feature_importances_


# In[35]:

cancer.feature_names


# In[37]:

cancer.data.shape[1]


# In[46]:

plt.figure(figsize = (10,10))
plt.barh(cancer.feature_names, tree.feature_importances_)


# In[51]:

import numpy as np

df = pd.DataFrame({'key1' : ['a','a','b','b','a'], 
                   'key2' : ['one','two','one','two','one'], 
                   'data1' : np.random.randn(5), 
                   'data2' : np.random.randn(5)})


# In[52]:

df


# In[53]:

grouped = df['data1'].groupby(df['key1'])


# In[54]:

grouped


# In[56]:

grouped.max()


# In[57]:

df['data1']


# In[58]:

df[['data1']]


# In[61]:

means = df['data1'].groupby([df['key1'], df['key2']]).mean()


# In[62]:

means


# In[70]:

means.unstack('key1')


# In[71]:

df


# In[73]:

df.groupby('key1').mean()


# In[74]:

df.groupby(['key1','key2']).size()


# In[75]:

for name, group in df.groupby('key1') : 
    print(name)
    print(group)


# In[85]:

dict(list(df.groupby('key1')))


# In[91]:

dict(list(df.groupby(df.dtypes, axis =1 )))


# In[92]:

df


# In[101]:

df.groupby(['key1','key2'])[['data2']].mean()


# In[104]:

df[['data2']].groupby([df['key1'],df['key2']]).mean()


# In[105]:

people = pd.DataFrame(sp.random.randn(5,5), 
                     columns = ['a','b','c','d','e'],
                    index = ['joe','steve','wes','jim','travis'])


# In[106]:

people


# In[108]:

people.ix[2:3, ['b','c']] = sp.nan


# In[109]:

people


# In[113]:

mapping = { 'a': 'red','b':'red','c':'blue', 'd':'blue', 'e':'red', 'f':'orange'}


# In[114]:

mapping


# In[115]:

by_column = people.groupby(mapping, axis =1)


# In[116]:

by_column.sum()


# In[117]:

people


# In[120]:

people.groupby(len).sum()


# In[121]:

df


# In[123]:

grouped = df.groupby('key1')


# In[124]:

grouped['data1'].quantile(0.9)


# In[126]:

grouped.describe().stack()


# In[127]:

df


# In[129]:

-0.657556 -  -1.474630


# In[130]:

def a(arr) :
    return arr.max() - arr.min()


# In[131]:

grouped.agg(a)


# In[132]:

df


# In[133]:

k1_means = df.groupby('key1').mean().add_prefix('mean_')


# In[134]:

k1_means


# In[138]:

pd.merge(df, k1_means, left_on= 'key1',right_index= True)


# In[139]:

df


# In[140]:

key = [ 'one' ,' two', 'one','two','one']


# In[141]:

people.groupby(key).mean()


# In[142]:

people


# In[145]:

from pandas import DataFrame, Series
s= Series(sp.random.randn(6))


# In[146]:

s


# In[147]:

s[::2] = sp.nan


# In[150]:

s.mean()


# In[149]:

s.fillna(s.mean())


# In[151]:

states = ['ohio','new york','vermont','florida','oregon','nevada','california','idaho']


# In[152]:

group_key = ['east'] * 4 + ['west'] * 4 


# In[153]:

data = Series(np.random.randn(8), index = states) 


# In[154]:

data[['vermont','nevada','idaho']] = sp.nan


# In[155]:

data


# In[156]:

data.groupby(group_key).mean()


# In[158]:

data.groupby(group_key).apply(lambda x : x.fillna(x.mean()))


# In[161]:

tips = pd.read_csv("C:/Users/SS/Desktop/P00000001-ALL.csv")
tips


# In[165]:

tips.pivot_table(index = ['smoker','day'], columns = 'size',margins = True)


# In[166]:

fec = pd.read_csv("C:/Users/SS/Desktop/P00000001-ALL.csv")


# In[167]:

fec


# In[168]:

fec.info()


# In[169]:

fec.ix[123456]


# In[175]:

unique_cands = fec.cand_nm.unique()


# In[176]:

unique_cands


# In[179]:

(fec.contb_receipt_amt > 0).value_counts()


# In[180]:

fec = fec[fec.contb_receipt_amt > 0]


# In[181]:

fec


# In[182]:

fec.cand_nm.isin(['Obama, Barack', 'Romney, Mitt'])


# In[194]:

a = Series([1,2,3])
b= Series([1,2,4])


# In[195]:

a.isin([1,2,4])


# In[196]:

by_occupation = fec.pivot_table('contb_receipt_amt', index = 'contbr_occupation', columns = 'party', aggfunc = 'sum')


# # 비지도 학습과 데이터 전처리

# In[199]:

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

x_train , x_test , y_train , y_test = train_test_split(cancer.data, cancer.target, random_state =1 )

print(x_train.shape)
print(x_test.shape)


# In[200]:

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)


# In[201]:

x_train_scaled = scaler.transform(x_train)


# In[235]:

fig , axes = plt.subplots(15,2 , figsize= (20,20))

k=0

for i in range(15) :
    for j in range(2): 
        
        axes[i,j].hist(cancer.data[:,k], bins = 50, label = cancer.target)
       
        k = k+1
        
fig.tight_layout()
plt.legend()


# In[264]:

fig , axes = plt.subplots(15,2 , figsize= (20,20))

k = 0
 
for i in range(15) :
    for j in range(2): 

        axes[i,j].hist(malignant[:, k], bins = 50 ,  color = 'g' , label = 'mal')
        axes[i,j].hist(benign[:, k],  bins = 50 ,color = 'r' , label = 'be')
        
        
        k = k+1

plt.legend(loc = 'best')  
fig.tight_layout()


# In[252]:

malignant = cancer.data[cancer.target == 1 ]


# In[253]:

benign = cancer.data[cancer.target == 0 ]


# In[243]:

malignant


# In[244]:

benign


# In[267]:

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
cancer = load_breast_cancer()


StandardScaler().fit(cancer.data)
x_scaled = scaler.transform(cancer.data)


# In[276]:

x_scaled.shape


# In[293]:

from sklearn.decomposition import PCA

pca = PCA(n_components =2)
pca.fit(x_scaled) 

x_pca = pca.transform(x_scaled)


# In[282]:

x_pca.shape


# In[342]:

plt.figure(figsize = (10,10))
plt.scatter( x_pca[:,0], x_pca[:,1], c= cancer.target)
plt.legend()
plt.xlabel('1st component')


# In[355]:

plt.figure(figsize = (10,10))
plt.scatter(x_pca[:,0][cancer.target ==0],x_pca[:,1][cancer.target ==0], label= 'a')
plt.scatter(x_pca[:,0][cancer.target ==1],x_pca[:,1][cancer.target ==1], label ='b')
plt.legend()


# In[357]:

pca.components_.shape


# In[358]:

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

x,y = make_blobs(random_state = 1)

kmeans = KMeans(n_clusters =3 )
kmeans.fit(x)


# In[359]:

kmeans.labels_


# In[361]:

x.shape


# In[362]:

y.shape


# In[364]:

kmeans.predict(x)


# In[370]:

kmeans = KMeans(n_clusters=3).fit(x)
plt.scatter(x[:,0], x[:,1], c = kmeans.labels_)


# In[371]:

kmeans.cluster_centers_


# In[373]:

from scipy.cluster.hierarchy import dendrogram, ward ,complete

x,y = make_blobs(random_state = 0 , n_samples = 50)

linkage_array = ward(x)
dendrogram(linkage_array)


# In[374]:

linkage_array = complete(x)
dendrogram(linkage_array)


# In[375]:

x,y = mglearn.datasets.make_wave(n_samples= 100)


# In[377]:

x.shape


# In[379]:

y.shape


# In[380]:

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 10 , include_bias = False)
poly.fit(x)
x_poly = poly.transform(x)


# In[382]:

x_poly.shape


# In[384]:

poly.get_feature_names()


# In[385]:

rnd = np.random.RandomState(0)
x_org = rnd.normal(size = (1000,3))
w = rnd.normal(size =3)


# In[391]:

rnd.normal(size = 3)


# In[399]:

x = rnd.poisson(10* np.exp(x_org))
y = np.dot(x_org, w )


# In[396]:

x.shape


# In[400]:

plt.hist(x, bins =10)


# In[406]:

plt.hist(np.log(x+1), bins =10)


# In[416]:

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
logreg = LogisticRegression()

scores = cross_val_score(logreg, iris.data, iris.target ,cv = 3)


# In[418]:

scores


# In[414]:

from sklearn.model_selection import cross_val_predict

iris = load_iris()
logreg = LogisticRegression()
scores = cross_val_predict(logreg, iris.data, iris.target ,cv = 5)


# In[415]:

scores


# In[420]:

from sklearn.model_selection import KFold
kfold = KFold(n_splits = 3)

cross_val_score(logreg, iris.data, iris.target, cv = kfold)


# In[422]:

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits =3)

cross_val_score(logreg , iris.data, iris.target, cv = kfold)


# In[423]:

from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv = loo)


# In[426]:

scores.mean()


# In[ ]:




# In[ ]:



