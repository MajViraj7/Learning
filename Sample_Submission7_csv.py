#!/usr/bin/env python
# coding: utf-8

# ## Importing Necessary Library

# In[26]:


import numpy as np 
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import metrics
 


# ## Reading csv File

# In[27]:


train_data=pd.read_csv("C:\\Users\\p\\Desktop\\data_train.csv")
test_data=pd.read_csv("C:\\Users\\p\\Desktop\\data_test.csv")


# ## Data Inspection

# In[28]:


train_data.info()


# In[29]:


test_data.info()


# ## Converting -1 value to nan

# In[30]:


train_data.replace(-1,np.nan,inplace=True)
test_data.replace(-1,np.nan,inplace=True)


# In[31]:


train_data.isnull().sum()


# In[32]:


test_data.isnull().sum()


# In[33]:


train_data.info()


# # Handling Missing values

# In[34]:


train_data = train_data.apply(lambda x: x.fillna(x.mean()),axis=0)
test_data=test_data.apply(lambda y: y.fillna(y.mean()),axis=0)
   
   
    
    
    
   


# ## Converting float to int

# In[35]:


train_data=(train_data.round(0).astype(int))
test_data=(test_data.round(0).astype(int))


# In[36]:


x1=train_data.drop("target",axis=1)
y1=train_data.target


# In[37]:


train_data.isnull().sum()


# ## Splitting Data into Train and Test

# In[38]:


X_train,X_test,Y_train,Y_test=train_test_split(x1,y1,test_size=0.2,random_state=4)


# In[39]:


train_data.info()


# ## Finding all Cateogorical Column

# In[40]:


cat_cols=X_train.loc[:,['cat' in i for i in X_train.columns]]
cat_cols


# ## Column Transform

# In[41]:


ct=ColumnTransformer(transformers=([
    ("step1",OneHotEncoder(sparse=False,handle_unknown='ignore'),["ind_02_cat","ind_04_cat","ind_05_cat","car_01_cat",
                                                                "car_02_cat","car_03_cat","car_04_cat","car_05_cat","car_06_cat",
                                                                "car_07_cat","car_08_cat","car_09_cat","car_10_cat","car_11_cat"])]),remainder="passthrough")


# ## Pipeline

# In[42]:


p = Pipeline([
    ('coltf_step',ct),
    ('model',DecisionTreeRegressor())
    
])


# ## Prediction

# In[43]:


p.fit(X_train,Y_train)



v=p.predict(X_test)


# In[44]:


v


# In[45]:


print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, v))


# In[46]:


prediction=p.predict(test_data)


# In[47]:


prediction


# In[48]:


import csv
with open('./sample_submission.csv', 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            

            filewriter.writerow(['id','target'])
            for i in range(len(test_data)):
                filewriter.writerow([test_data['id'][i],prediction[i]])


# In[49]:


sample_submission_csv=pd.read_csv("./sample_submission.csv")
sample_submission_csv


# In[50]:


sample_submission_csv.to_csv("Sample_Submission7_csv.csv",index=False)

