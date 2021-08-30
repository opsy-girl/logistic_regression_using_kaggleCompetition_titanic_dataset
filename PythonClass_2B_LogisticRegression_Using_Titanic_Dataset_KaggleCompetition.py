#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This cell is to install and import my packages
#!pip install pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


# To read my data file.
TitaniTrain = pd.read_csv(r'C:\Users\XXX\Downloads\titanic\TitanicTrain.csv')


# In[3]:


#Check summary information on data
TitaniTrain.info()


# In[4]:


#Observe the 'Survived' column values
TitaniTrain['Survived'].unique()


# In[5]:


#Observe the 'Pclass' column values
TitaniTrain['Pclass'].unique()


# In[6]:


#Observe the 'Sex' column values
TitaniTrain['Sex'].unique()


# In[7]:


#Observe first few values in dataset
TitaniTrain.head()


# In[8]:


#Check summary information of table
TitaniTrain.info()


# In[9]:


#Observe the distinct values in 'Survived' column of dataset
TitaniTrain['Survived'].unique()


# In[10]:


#Observe the distinct values in 'Pclass' column of dataset
TitaniTrain['Pclass'].unique()


# In[11]:


#Observe the distinct values in 'Sex' column of dataset
TitaniTrain['Sex'].unique()


# In[12]:


#Convert Ages to months for better precision
TitaniTrain['Age'] = TitaniTrain['Age'] * 12


# In[13]:


#Convert Ages in months to whole numbers for better precision
TitaniTrain['Age'] = TitaniTrain['Age'].round()


# In[14]:


TitaniTrain['Age'].unique()


# In[15]:


#TitaniTrain['Age'].unique()
print('Max Age is ', TitaniTrain['Age'].max(), 'and min age is ', TitaniTrain['Age'].min())


# In[16]:


#Check for records with missing age values
TitaniTrain[TitaniTrain['Age'].isnull()==True]['Age']


# In[17]:


#Extract mean age value
TitaniTrain['Age'].mean()  #356


# In[18]:


# Extract the index of records with missing age values. The index will be used in a loop for replacing the missing values

AgeNull = pd.DataFrame(TitaniTrain['Age'][TitaniTrain['Age'].isnull()==True])
AgeNullList = AgeNull.index.tolist()  
AgeNullList


# In[19]:


# Run a loop to replace only missing values with the mean age value

for m in AgeNullList:
    TitaniTrain['Age'][m] = 356


# In[20]:


#Check if there are still any missing values in age column
TitaniTrain[TitaniTrain['Age'].isnull()==False]['Age']


# In[21]:


#import libraries for plotting the normalization curve
import scipy.stats as stats
import pylab as pl


# In[22]:


#Sort Age values to observe better
sorted(np.array(TitaniTrain[TitaniTrain['Age'].isnull()==False]['Age']))


# In[23]:


#Check the skewness of the Age column using the normalization plot or historam plot

normValues = sorted(np.array(TitaniTrain[TitaniTrain['Age'].isnull()==False]['Age']))
fit = stats.norm.pdf(normValues, np.mean(normValues), np.std(normValues))
pl.plot(normValues,fit, '-o')
#pl.hist(normValues)
pl.show()


# In[24]:


#Extract the mean, standard deviation and median values of the Age column
print(np.mean(normValues).round(),", " , np.std(normValues).round(), ", ", np.median(normValues))


# In[25]:


#COnvert the string labeled columns that you want to use in the model to integer values using the index values instead of string
#That is convert string category columns to index or integer values

TitaniTrain['Sex']=TitaniTrain['Sex'].astype('category')

TitaniTrain["Sex"] = TitaniTrain["Sex"].cat.codes


# In[26]:


#That is convert string category columns to index or integer values

TitaniTrain['Embarked']=TitaniTrain['Embarked'].astype('category')
TitaniTrain["Embarked"] = TitaniTrain["Embarked"].cat.codes


# In[27]:


len(TitaniTrain['Ticket'].unique()) #type of ticket or ticket no is useless for insight


# In[28]:


# Drop index columns from dataset. Or do not include any form of index columns in the ML models as it may not give any insight

TitaniTrain=TitaniTrain.drop(['PassengerId','Cabin'], axis = 1)
print('')


# In[29]:


#Check the significance of each independent variable to the dependent variable
TitaniTrain.corr()['Survived']


# In[30]:


#Assign variables to the x and y series in preparation for the machine learning model
x = TitaniTrain[['Sex', 'Pclass', 'Fare',  'Embarked', 'Age', 'SibSp','Parch',]]
y= TitaniTrain['Survived']


# In[31]:


from sklearn.linear_model import LogisticRegression


# In[32]:


model = LogisticRegression()


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# In[35]:


#Fit the model to the data

model.fit(x_train, y_train)


# In[36]:


#Check model predictions on test data

predictions = model.predict(x_test)


# In[37]:


#Check the model accuracy score
score = model.score(x_test, y_test)
print(score)


# In[ ]:




