#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
from sklearn import tree
from  sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


# In[6]:


#Loading the file
df = pd.read_excel(r'D:\Books\Project\Draft1_fabien\leak-exp-day2\Exact\release2a.xlsx')
s1 = df.iloc[:,0]
s2 = df.iloc[:,1]
s3 = df.iloc[:,2]
s4 = df.iloc[:,3]
s5 = df.iloc[:,4]
s6 = df.iloc[:,5]
s7 = df.iloc[:,6]
s8 = df.iloc[:,7]
s9 = df.iloc[:,8]
s10 = df.iloc[:,9]
s11 = df.iloc[:,10]
s12 = df.iloc[:,11]
s13 = df.iloc[:,12]
s14 = df.iloc[:,13]
s15 = df.iloc[:,14]
s16 = df.iloc[:,15]
s17 = df.iloc[:,16]
s18 = df.iloc[:,17]
s19 = df.iloc[:,18]
s20 = df.iloc[:,19]
status = df.iloc[:,20] 


# In[7]:


#creating features and labels
n_features = list(zip(s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20))
n_samples = status


# In[8]:


#Decision tree  regression
clf = tree.DecisionTreeRegressor()
#spliting of data
X_train, X_test, y_train, y_test = train_test_split(n_features,n_samples, test_size=0.5,random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)
#train model
clf.fit(X_train,y_train)
#prediction
y_pred = clf.predict(X_test_std)
print('percentage Accuracy:',100*metrics.accuracy_score(y_test,y_pred))


# In[9]:


#false prediction
print((y_test != y_pred).sum(),'/',((y_test==y_pred).sum()+(y_test != y_pred).sum()))


# In[22]:


#Graph ploting Features vs Predict values
k = []
for i in range(0,len(X_test)):
    k.append(i+1)
plt.figure(figsize = (20,8))
plt.xlim(0,150)
sns.lineplot(y = y_pred, x = k,label = "y_pred",color = 'red')
sns.lineplot(y = y_test, x = k,label ="y_test",color = 'blue')
plt.legend()
plt.show()


# In[29]:


#3d plot
fig = plt.figure(figsize = (20,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y_pred, y_test,k, c='r', marker='o')
ax.set_xlabel('Predicted Value')
ax.set_ylabel('Test Value')
ax.set_zlabel('Feature Index')

plt.show()


# In[57]:


plt.figure(figsize = (20,8))
plt.plot(k, y_pred, c='r', label='y_pred',linestyle = 'dotted',linewidth = 6.0)
plt.plot(k, y_test, c='b', label='y_test')
plt.xlim(40,120)
plt.legend()
plt.show()


# In[ ]:




