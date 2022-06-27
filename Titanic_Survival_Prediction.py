#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


from sklearn.model_selection import train_test_split


# In[3]:


from sklearn.linear_model import LogisticRegression


# In[4]:


from sklearn.metrics import accuracy_score


# Data Collection and processing

# In[5]:


# load data from csv file to pandas DataFrame


# In[6]:


titanic_data=pd.read_csv("C://Users//simran sehgal//OneDrive//Desktop//train.csv")


# In[7]:


# Printing the first 5 rows of our dataframwe
titanic_data.head()


# In[8]:


# number of rows and columns
titanic_data.shape


# In[9]:


# getting some information about the data
titanic_data.info()


# In[10]:


# check missing value in each column
titanic_data.isnull().sum()


# Handling missing values
# 

# In[11]:


# drop the cabin columns from dataframe
titanic_data = titanic_data.drop(columns='Cabin',axis=1)


# In[12]:


# replacingthe missing values in"Age" with mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)


# In[13]:


# finding the mode value of the embarked column
print(titanic_data['Embarked'].mode()[0])


# In[14]:


# replaceing the missing value in "Embarked" column with the mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0],inplace=True)


# In[15]:


titanic_data.isnull().sum()


# Data Analysis

# In[16]:


titanic_data.describe()


# In[17]:


# finding the number of people survied and not suvived
titanic_data['Survived'].value_counts()


# Data Visualization

# In[18]:


sns.set()


# In[19]:


# making a count plot for "survived" column
sns.countplot('Survived',data=titanic_data)


# In[20]:


# making a count plot for "sex" column
sns.countplot('Sex',data=titanic_data)


# In[21]:


# number of survivors gender wise
sns.countplot('Sex',hue='Survived',data=titanic_data)


# In[22]:


# making a countplot for 'Pclass'
sns.countplot('Pclass',data=titanic_data)


# In[23]:


sns.countplot('Pclass',hue='Survived',data=titanic_data)


# Encoding the Categorical Columns

# In[24]:


titanic_data['Sex'].value_counts()


# In[25]:


titanic_data['Embarked'].value_counts()


# In[26]:


#converting categorical columns
titanic_data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)


# In[27]:


titanic_data.head()


# Separating Features and Targets

# In[28]:


X=titanic_data.drop(columns=['PassengerId','Name','Ticket','Survived'],axis=1)
Y=titanic_data['Survived']


# Splitting the data into Train and Test Set

# In[29]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2, random_state=2)


# In[30]:


print(X.shape,X_train.shape,X_test.shape)


# Model Training

# Logistic Regression 

# In[31]:


model = LogisticRegression()


# In[32]:


# training the logistic Regression model with training data
model.fit(X_train,Y_train)


# In[33]:


# accuracy on training data
X_train_prediction=model.predict(X_train)


# In[34]:


print(X_train_prediction)


# In[37]:


training_data_accuracy=accuracy_score(Y_train,X_train_prediction)


# In[38]:


print('Accuracy score of training data : ', training_data_accuracy)


# In[39]:


# accuracy on test data
X_test_prediction=model.predict(X_test)


# In[40]:


print(X_test_prediction)


# In[43]:


test_data_accuracy=accuracy_score(Y_test,X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)


# In[44]:


training_data_accuracy=accuracy_score(Y_test,X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)


# In[ ]:


# According to the data female Passenger survived more and 3rd class passenger survived more

