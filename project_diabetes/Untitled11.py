#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
plt.style.use("fivethirtyeight")

import warnings
warnings.filterwarnings("ignore")


# In[7]:


pd.read_csv(r"D:/Downloads/archive (13)/diabetes.csv")


# #EXPLORE DATA

# In[8]:


pf.head()


# In[9]:


pf.info()


# In[10]:


pf.describe()


# In[13]:


pf.duplicated().sum()


# #ANALYSIS

# In[14]:


pf.corr()


# In[19]:


sns.heatmap(pf.corr(),annot=True,fmt='.1f',linewidth=.5)


# In[22]:


sns.countplot(x="Outcome", data= pf, palette=['g','r'])


# In[23]:


plt.figure(figsize=(20,6))

plt.subplot(1,3,1)
plt.title("Counter Plot")
sns.countplot(x="Pregnancies", data=pf)

plt.subplot(1,3,2)
plt.title("Distribution Plot")
sns.distplot(pf["Pregnancies"])

plt.subplot(1,3,3)
plt.title("Box Plot")
sns.boxplot(x=pf["Pregnancies"])

plt.show()


# In[25]:


sns.boxplot(pf.Age)


# #CREATE MODEL

# In[26]:


x=pf.drop('Outcome',axis=1)
y=pf['Outcome']


# In[28]:


x_train ,x_test , y_train , y_test = train_test_split(x,y,test_size=.2)


# model1= LogisticRegression()
# model1.fit(x_train , y_train)

# In[58]:


model1=LogisticRegression()
model2=SVC()
model3=RandomForestClassifier()
model4=GradientBoostingClassifier(n_estimators=1000)


# In[93]:


columns=['LogisticRegression','SVC','RandomForestClassifier','GradientBoostingClassifier']
result1=[]
result2=[]
result3=[]


# In[94]:


def cal(model):    
    model.fit(x_train , y_train)
    pre = model.predict(x_test)
    accuracy = accuracy_score(pre,y_test)
    recall = recall_score(pre,y_test)
    f1 = f1_score(pre,y_test)
    conf_matrix = confusion_matrix(pre,y_test)
    
    result1.append(accuracy)
    result2.append(recall)
    result3.append(f1)
    
    sns.heatmap(confusion_matrix(pre,y_test),annot=True)
    print(model)
print("accuracy is :" ,accuracy , "recall is :" ,recall , "f1 is :" ,f1)
cal(model1)


# In[95]:


cal(model2)


# In[96]:


cal(model4)


# In[97]:


cal(model3)


# In[98]:


result1


# In[99]:


result2


# In[100]:


result3


# In[104]:


FinalResult=pd.DataFrame({
    'Algorithm': columns,
    'Accuracies': result1,
    'Recall': result2,
    'FScore': result3
})


# In[105]:


fig, ax = plt.subplots(figsize=(20,5))
plt.plot(FinalResult.Algorithm, result1, label='Accuracy')
plt.plot(FinalResult.Algorithm, result2, label='Recall')
plt.plot(FinalResult.Algorithm, result3, label='F1Score')
plt.legend()
plt.show()


# In[ ]:




