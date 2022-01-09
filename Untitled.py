#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv(r"/home/arunsundar/Downloads/Final_SolarData.csv")


# In[4]:


data['batterypower']


# In[13]:


load1=list(data['load_power1'])


# In[15]:


load2=list(data['load_power2'])


# In[28]:


load=[]


# In[29]:


for i in range(0,len(load1)):
    load.append(((load1[i]+load2[i])/2))


# In[30]:


load


# In[31]:


data['loadpower']=load


# In[32]:


data=[]
for i in data['date_time']:
    


# In[66]:


t=[]


# In[67]:


for i in data['date_time']:
    s=i.split(" ")
    t.append(s[1])


# In[73]:


data['time']=t


# In[85]:


data1=data[data['time']=="23:59"]


# In[49]:


data.columns


# In[82]:


for i in 


# In[86]:


data1


# In[98]:


result=[]
lp=list(data1['loadpower'])
bp=list(data1['batterypower'])
for j in range(0,len(lp)):
    if (bp[j]>lp[j]):
        result.append(1)
    else:
        result.append(0)
    
    
    


# In[101]:


month=[]
date=[]
for i in data1['date']:
    s=i.split('-')
    month.append(s[1])
    date.append(s[2])
    


# In[103]:


dic={"Date":date,"Month":month,"Result":result}


# In[104]:


org=pd.DataFrame(dic)


# In[105]:


org


# In[109]:





# In[110]:


plt.plot(data['date'],data['loadpower'])


# In[111]:


plt.plot(data['date'],data['batterypower'])


# In[112]:


pip install sklearn


# In[113]:


from sklearn.model_selection import train_test_split


# In[136]:


x=org.drop(labels='Result',axis=1)


# In[140]:


y=org['Result']


# In[141]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=88,shuffle=True,stratify=y)


# In[142]:


x_train


# In[143]:





# In[144]:


knn=KNeighborsClassifier(n_neighbors=6, metric='minkowski',p=1)


# In[145]:


knn.fit(x,y)


# In[158]:





# In[156]:





# In[157]:





# In[162]:


rid=input("dd-mm-yyy")
re=rid.split("-")
d={"Date":[re[0]],"Month":[re[1]]}
u=pd.DataFrame(d)
knn.predict(u)


# In[ ]:




