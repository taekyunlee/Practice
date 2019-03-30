
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


# In[28]:

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

x,y = make_blobs(random_state=1)

kmeans = KMeans(n_clusters=5)
kmeans.fit(x)


# In[29]:

kmeans.labels_


# In[30]:

kmeans.cluster_centers_


# In[16]:

kmeans.inertia_


# In[18]:

x.shape


# In[22]:

x


# In[36]:

plt.scatter(x[:,0], x[:,1], c = kmeans.labels_ )


# In[49]:

la = pd.DataFrame(kmeans.labels_)
la.rename(columns ={0: 'la'}, inplace= True)


# In[50]:

dat = pd.DataFrame(x)


# In[52]:

data = pd.concat([dat,la], axis = 1 )


# In[64]:

data['la'] = data['la'].astype(str)


# In[79]:

plt.scatter(data[data['la'] == '0'][0],data[data['la'] == '0'][1] )
plt.legend()


# In[83]:

data[data['la'] == '0'][1]


# In[6]:

import sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

x,y = make_blobs(random_state = 1)

agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(x)



# In[14]:

from scipy.cluster.hierarchy import dendrogram, ward
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

x, y = make_blobs(random_state= 0 , n_samples=12)

linkage_array = ward(x)

dendrogram(linkage_array)

ax = plt.gca()
bounds = ax.get_xbound()

ax.plot(bounds)


# In[ ]:



