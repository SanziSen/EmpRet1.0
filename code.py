#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[6]:


df = pd.read_csv("HR_comma_sep.csv")
df


# In[7]:


plt.bar(df.salary, df.left, align='center', alpha=0.5)
plt.ylabel('Left')
plt.title('Salary')
plt.show()


# In[8]:


plt.bar(df.sales, df.left, align='center', alpha=0.5)
plt.ylabel('Left')
plt.title('Department')
plt.show()


# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.satisfaction_level,df.left, test_size=0.2)


# In[11]:


from sklearn.linear_model import LogisticRegression


# In[12]:


model=LogisticRegression(solver='lbfgs')


# In[13]:


df.shape


# In[14]:


X_train


# In[15]:


y_train


# In[26]:


n=np.array(X_train).reshape(-1,1)


# In[28]:


n.shape


# In[29]:


model.fit(n,y_train)


# In[31]:


no=np.array(X_test).reshape(-1,1)


# In[ ]:





# In[33]:


model.score(no,y_test)


# In[ ]:




