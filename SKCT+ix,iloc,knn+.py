
# coding: utf-8

# In[6]:

import pandas as pd
from pandas import Series


# In[12]:

obj = Series([4,5,-5,3])


# In[13]:

obj


# In[15]:

obj.values


# In[17]:

obj.index


# In[18]:

obj2 = Series([4,7,-5,3], index = ['d','b','a','c'])


# In[19]:

obj2


# In[20]:

obj2.index


# In[21]:

obj2.values


# In[22]:

obj2['a']


# In[24]:

obj2[['a','c']]


# In[25]:

obj2['a'] =1 


# In[26]:

obj2


# In[29]:

obj2[obj2>2]


# In[30]:

obj2*2


# In[31]:

obj2


# In[33]:

obj2+1


# In[35]:

import numpy as np


# In[39]:

np.exp(1)


# In[40]:

'b' in obj2


# In[41]:

'e' in obj2


# In[42]:

sdata = {'Ohio' : 35000, 'Texas' : 71000, 'Oregon' : 16000, 'Utah' : 5000}


# In[43]:

sdata


# In[44]:

obj3 = Series(sdata)


# In[45]:

obj3


# In[46]:

states = ['California', 'Ohio','Oregon','Texas'] 


# In[47]:

obj4 = Series(sdata, index = states)


# In[48]:

obj4


# In[50]:

pd.isnull(obj4)


# In[51]:

pd.notnull(obj4)


# In[52]:

obj4.isnull()


# In[54]:

obj4.name = 'population'


# In[56]:

obj4.name


# In[57]:

obj4.index.name = 'state'


# In[58]:

obj4


# In[59]:

data = {'state' :['ohio','ohio','ohio','navada','nevada'] ,
        'year' : [2000,2001,2002,2001,2002], 
        'pop' : [1.5, 1.7, 3.6, 2.4, 2.9]}


# In[60]:

data


# In[62]:

frame = pd.DataFrame(data)


# In[63]:

frame


# In[64]:

data


# In[66]:

pd.DataFrame(data, columns = ['year','state','pop'])


# In[67]:

frame2 = pd.DataFrame(data, columns = ['year','state','pop','debt'], index = ['one','two','three','four','five'])


# In[68]:

frame2


# In[69]:

frame2.columns


# In[70]:

frame2.index


# In[71]:

frame2['state']


# In[73]:

frame2.state


# In[75]:

frame2.ix['three']


# In[77]:

frame2


# In[78]:

frame2['debt'] = 16.5


# In[79]:

frame2


# In[80]:

frame2['debt'] = np.arange(5)


# In[81]:

frame2


# In[82]:

val = Series([-1.2,-1.5, -1.7] , index = ['two','four','five'])


# In[83]:

frame2['debt'] = val


# In[84]:

frame2


# In[85]:

frame2['eastern'] = frame2.state == 'ohio'


# In[86]:

frame2


# In[87]:

del frame2['eastern']


# In[88]:

frame2


# In[89]:

frame2.columns


# In[90]:

pop = {'nevada' : {2001 :2.4, 2002:2.9}, 'ohio' : {2000 :1.5,2001:1.7, 2002:3.6}}


# In[91]:

pop


# In[92]:

frame3 = pd.DataFrame(pop)


# In[93]:

frame3


# In[95]:

frame3.T


# In[96]:

pd.DataFrame(pop, index = [2001,2002,2003])


# In[97]:

pdata = {'ohio' :frame3['ohio'][:-1] , 
         'nevada' : frame3['nevada'][:2]}


# In[98]:

pdata


# In[106]:

frame3['ohio'][: -1]


# In[107]:

frame3


# In[108]:

frame3.index.name = 'year'


# In[110]:

frame3.columns.name = 'state'


# In[111]:

frame3


# In[113]:

frame3.values


# In[115]:

obj = Series(range(3) , index = ['a','b','c'])


# In[116]:

obj


# In[117]:

index = obj.index


# In[118]:

index


# In[119]:

index[1:]


# In[120]:

pd.Index(np.arange(3))


# In[121]:

frame3


# In[122]:

'ohio' in frame3.columns


# In[123]:

2003 in frame3.index


# In[124]:

obj


# In[125]:

obj = Series([4.5, 7.2, -5.3, 3.6], index = ['d','b','a','c'])


# In[126]:

obj


# In[127]:

obj2 = obj.reindex(['a','b','c','d','e'])


# In[128]:

obj2


# In[129]:

obj.reindex(['a','b','c','d','e'], fill_value =0) 


# In[133]:

frame = pd.DataFrame(np.arange(9).reshape((3,3)), index = ['a','c','d'], 
                  columns = ['ohio', 'texas','california'])


# In[134]:

frame


# In[135]:

frame2 = frame.reindex(['a','b','c','d'])


# In[136]:

frame2


# In[139]:

states = ['texas','utah','california']
frame.reindex(columns = states)


# In[140]:

frame


# In[141]:

frame.ix[['a','b','c','d'], states]


# In[142]:

obj = Series(np.arange(5.), index = ['a','b','c','d','e'])


# In[143]:

obj


# In[144]:

new_obj = obj.drop('c')


# In[145]:

new_obj


# In[146]:

obj.drop(['d','c'])


# In[147]:

data = pd.DataFrame(np.arange(16).reshape((4,4)), 
                    index = ['ohio','colorado','utah','new york'],
                    columns = ['one','two','three','four'])


# In[148]:

data


# In[149]:

data.drop(['colorado','ohio'])


# In[150]:

data.drop('two', axis =1 )


# In[154]:

data.drop(['two','four'] , axis = 1) 


# In[155]:

obj = Series(np.arange(4.), index = ['a','b','c','d'])


# In[156]:

obj


# In[157]:

obj['b']


# In[158]:

obj[1]


# In[159]:

obj[0]


# In[160]:

obj[2:4]


# In[161]:

obj[['b','a','d']]


# In[163]:

obj[[1,3]]


# In[164]:

obj


# In[165]:

obj[obj < 2]


# In[167]:

obj['b':'c'] = 5


# In[168]:

obj


# In[170]:

data = pd.DataFrame(np.arange(16).reshape((4,4)), index = ['ohio','colorado','utah','new york'] , 
                columns = ['one','two','three','four'])


# In[172]:

data


# In[173]:

data['two']


# In[174]:

data[['three','one']]


# In[175]:

data[ : 2]


# In[176]:

data[data['three'] >5 ]


# In[177]:

data


# In[178]:

data < 5


# In[179]:

data[data<5] = 0


# In[180]:

data


# In[181]:

data


# In[182]:

data.ix['colorado', ['two','three']]


# In[183]:

data.ix['colorado', [3,0,1]]


# In[184]:

data.ix[2]


# In[185]:

data.ix[ : 'utah', 'two']


# In[186]:

data.ix[data.three >5 , : 3]


# In[187]:

data


# In[189]:

arr = np.arange(12.).reshape((3,4))


# In[190]:

arr


# In[191]:

arr[0]


# In[192]:

arr[2]


# In[193]:

arr[1:2]


# In[195]:

frame = pd.DataFrame(np.arange(12.).reshape((4,3)), columns = list('bde'), 
                 index = ['utah','ohio','texas','oregon'])


# In[196]:

frame


# In[197]:

series = frame.ix[0]


# In[198]:

series


# In[199]:

frame - series 


# In[201]:

series2 = Series(range(3), index = ['b','e','f'])


# In[203]:

series2


# In[204]:

frame


# In[205]:

frame + series2


# In[206]:

series3 = frame['d']


# In[208]:

series3


# In[213]:

frame.sub(series3, axis = 0 )


# In[215]:

frame


# In[222]:

import scipy as sci


# In[219]:

frame = pd.DataFrame(np.random.randn(4,3), columns = list('bde'), index = ['utah', 'ohio','texas','oregon'])


# In[220]:

sci.absolute(frame)


# In[223]:

sci.absolute(frame)


# In[224]:

f = lambda x : x.max() - x.min()


# In[225]:

frame


# In[227]:

frame.apply(f, axis = 1 )


# In[228]:

frame


# In[229]:

obj = Series(range(4), index = ['d','a','b','c'])


# In[230]:

obj


# In[231]:

obj.sort_index()


# In[232]:

frame = pd.DataFrame(np.arange(8).reshape((2,4)), index = ['three','one'], columns = ['d','a','b','c'])


# In[233]:

frame


# In[234]:

frame.sort_index()


# In[235]:

frame.sort_index(axis =1 ) 


# In[236]:

frame.sort_index(ascending = False)


# In[237]:

obj = Series([4,7,-3,2])


# In[238]:

obj


# In[239]:

obj.sort_values()


# In[241]:

frame = pd.DataFrame({'b': [4,7,-3,2], 'a' : [0,1,0,1]})


# In[242]:

frame


# In[243]:

frame.sort_values(by = 'b')


# In[244]:

obj = Series([7,-5,7,4,2,0,4])


# In[245]:

obj


# In[246]:

obj.rank(ascending = False, method = 'max')


# In[247]:

obj.rank(method = 'first')


# In[250]:

obj = Series(range(5), index = ['a','a','b','b','c'])


# In[251]:

obj


# In[253]:

obj.index.is_unique


# In[254]:

obj['a']


# In[255]:

df = pd.DataFrame(np.random.randn(4,3), index = ['a','a','b','b'])


# In[256]:

df


# In[257]:

df.ix['b']


# In[259]:

df = pd.DataFrame([[1.4,np.nan],[7.1,-4.5],[np.nan,np.nan],[0.75,-1.3]], index = ['a','b','c','d'],
              columns = ['one','two'])


# In[260]:

df


# In[261]:

df.sum()


# In[262]:

df.sum(axis = 1 )


# In[263]:

df.mean(axis =1 , skipna = False)


# In[264]:

df


# In[266]:

df.idxmax()


# In[267]:

df.cumsum()


# In[268]:

df.describe()


# In[269]:

df


# In[270]:

obj = Series(['a','a','b','c'] * 4)


# In[271]:

obj


# In[272]:

obj.describe()


# In[280]:

df.quantile(0.7)


# In[274]:

df


# In[281]:

obj = Series(['c','a','d','a','a','b','b','c','c'])


# In[282]:

obj


# In[283]:

uniques = obj.unique()


# In[284]:

uniques


# In[286]:

uniques.sort()


# In[287]:

uniques


# In[288]:

obj.value_counts()


# In[292]:

pd.value_counts(obj.values, sort =False)


# In[290]:

obj.values


# In[294]:

mask = obj.isin(['b','c'])


# In[295]:

mask


# In[296]:

obj[mask]


# In[298]:

data = pd.DataFrame({'Qu1' : [1,3,4,3,4], 
                  'Qu2' : [2,3,1,2,3],
                  'Qu3' : [1,5,2,4,4]})


# In[299]:

data


# In[304]:

result = data.apply(pd.value_counts).fillna(0)


# In[305]:

result


# In[308]:

string_data = Series(['aardvark','artichoke', np.nan,'avocado']) 


# In[309]:

string_data


# In[310]:

string_data


# In[311]:

string_data.isnull()


# In[312]:

string_data[0] = None


# In[313]:

string_data


# In[314]:

string_data.isnull()


# In[315]:

from numpy import nan as NA


# In[316]:

data = Series([1,NA, 3.5, NA, 7])


# In[317]:

data


# In[318]:

data.dropna()


# In[319]:

data[data.notnull()]


# In[324]:

df = pd.DataFrame(np.random.randn(7,3))


# In[328]:

df.ix[ :4 ,1 ] = NA ; df.ix[:2,2] = NA


# In[329]:

df


# In[332]:

df.dropna(thresh =2)


# In[333]:

df


# In[334]:

df.fillna(0)


# In[335]:

df.fillna({1: 0.5, 3: -1})


# In[336]:

data = Series([1., NA, 3.5, NA, 7])


# In[337]:

data


# In[338]:

data.fillna(data.mean())


# In[339]:

data = Series(np.random.randn(10), index = [['a','a','a','b','b','b','c','c','d','d'], 
                                           [1,2,3,1,2,3,1,2,2,3]])


# In[340]:

data


# In[342]:

data.index


# In[343]:

data


# In[344]:

data.unstack()


# In[345]:

data.unstack().stack()


# In[348]:

frame = pd.DataFrame({'a':range(7), 'b':range(7,0,-1), 'c':['one','one','one','two','two','two','two'],
                     'd' : [0,1,2,0,1,2,3]})


# In[349]:

frame 


# In[350]:

frame2 = frame.set_index(['c','d'])


# In[351]:

frame2


# In[352]:

frame.set_index(['c'])


# In[353]:

frame


# In[354]:

ser3 = Series(range(3) , index = [-5 , 1, 3 ])


# In[355]:

ser3


# In[357]:

ser = Series(np.arange(3.))
ser.iloc[2] 


# In[358]:

ser


# In[360]:

frame = pd.DataFrame(np.arange(6).reshape((3,2)), index = [2,0,1])


# In[361]:

frame


# In[362]:

frame.iloc[0]


# In[363]:

frame.iloc[1]


# In[364]:

frame.iloc[2]


# In[366]:

import mglearn
x,y = mglearn.datasets.make_forge()


# In[367]:

x


# In[368]:

y


# In[384]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

mglearn.discrete_scatter(x[:,0],x[:,1] , y)
plt.legend(["class 0","class 1"] , loc = 'best')
plt.xlabel("1st feature")
plt.ylabel("2nd feature")


# In[382]:

x.shape


# In[385]:

print("X.shape : {}".format(x.shape))


# In[392]:

import sklearn
from sklearn.model_selection import train_test_split
x, y = mglearn.datasets.make_forge()
x_train , x_test , y_train, y_test = train_test_split(x,y, random_state = 0 )


# In[393]:

x


# In[394]:

y


# In[419]:

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 3)


# In[420]:

clf


# In[421]:

clf.fit(x_train, y_train)


# In[422]:

clf.predict(x_test)


# In[423]:

print("테스트 정확도 {}".format(clf.score(x_test, y_test)))


# In[424]:

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
x_train, x_test , y_train, y_test = train_test_split(cancer.data ,cancer.target, random_state =66) 


# In[425]:

cancer.data.shape 


# In[426]:

cancer.target.shape


# In[427]:

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1,11)


# In[428]:

list(range(1,11))


# In[429]:

for n_neighbors in neighbors_settings :
    
    clf = KNeighborsClassifier(n_neighbors= n_neighbors)
    clf.fit(x_train, y_train) 
    
    training_accuracy.append(clf.score(x_train,y_train))
    
    test_accuracy.append(clf.score(x_test, y_test ))
    
plt.plot(neighbors_settings, training_accuracy, label = 'train')
plt.plot(neighbors_settings, test_accuracy, label = 'test')
plt.ylabel("accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# In[434]:

x,y = mglearn.datasets.make_wave(n_samples = 40)
from sklearn.neighbors import KNeighborsRegressor
fig, axes = plt.subplots(1,3, figsize = (15,4))

line = np.linspace(-3,3,1000).reshape(-1,1)

for n_neighbors , ax in zip([1,3,9], axes) : 
    reg = KNeighborsRegressor(n_neighbors = n_neighbors)
    reg.fit(x_train, y_train) 
    ax.plot(line, reg.predict(line))
    ax.plot(x_train, y_train, '^', c= mglearn.cm2(0), markersize =8)
    ax.plot(x_test, y_test, 'v', c= mglearn.cm2(1), markersize = 8)
    


# In[ ]:



