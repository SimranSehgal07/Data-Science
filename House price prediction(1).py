#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import warnings
warnings.filterwarnings("ignore")


# In[66]:


import math


# In[67]:


from sklearn.model_selection import train_test_split


# In[68]:


import statsmodels.api as sm 


# In[69]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler


# In[70]:


from sklearn.linear_model import LinearRegression 


# In[71]:


import sklearn.metrics as metrics


# In[72]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[73]:


# Loading and Exploring the dataset
raw_data=pd.read_csv("C://Users//simran sehgal//OneDrive//Desktop//data.csv")


# In[74]:


# Quick look at the data
raw_data.head()


# In[75]:


raw_data.tail()


# In[76]:


raw_data.info()


# In[77]:


raw_data.shape


# In[78]:


raw_data.describe(include="all")


# In[79]:


# Let us check the accuracy of data without preprocessing
# Define the variables
x1=raw_data.drop(['price','date','street','city','statezip','country'], axis=1)
y=raw_data['price']


# In[80]:


# Add a constant esentially we are adding a new column( equal in lenght to x), which consist of only 1
x=sm.add_constant(x1)
#Fit the model, according to the OLS (ordinary least squares) method with a dependent variable y and independent variable x
results=sm.OLS(y,x).fit()
#Print a summary of model
results.summary()


# In[81]:


raw_data.isnull().sum()


# In[82]:


raw_data[raw_data==0].count()


# In[83]:


zero_price=raw_data[raw_data['price']==0]
zero_price.describe().T


# In[84]:


# In order to decide what to do with null values
# we need to find correlaion between the values
corr_matrix=raw_data.corr()
fig, ax=plt.subplots(figsize=(15,6))
sns.heatmap(corr_matrix, annot= True)


# In[85]:


# By looking at the median value of the sqft_living variable, which affects the price the most,
# I divided the prices that appear 0 into two groups. 
# Afterwards, I decided on the value that I would assign to the empty prices in these two groups 
# by looking at the median values of the 3 variables that most affected the price value.

low_price_data = raw_data[(raw_data['sqft_living'] < zero_price['sqft_living'].median()) &
         (raw_data['bathrooms'] < zero_price['bathrooms'].median()) &
         (raw_data['sqft_above'] < zero_price['sqft_above'].median()) ]
low_price = low_price_data.price.median()

high_price_data = raw_data[(raw_data['sqft_living'] > zero_price['sqft_living'].median()) &
         (raw_data['bathrooms'] > zero_price['bathrooms'].median()) &
         (raw_data['sqft_above'] > zero_price['sqft_above'].median()) ]
high_price = high_price_data.price.median()

data_prc = raw_data.copy()
data_prc['price'] = np.where(((data_prc['price']==0) & (data_prc['sqft_living'] > zero_price['sqft_living'].median())), high_price, data_prc.price) 
data_prc['price'] = np.where(((data_prc['price']==0) & (data_prc['sqft_living'] <= zero_price['sqft_living'].median())), low_price, data_prc.price)

data_prc.price[data_prc.price==0].count()


# In[86]:


# I will print the distrubution plots to decide 
# which method to use fill in the unknown zero values in the bedrooms and batromms columns.
# As you may notice, there is some skewness that will affect the mean of both features. 
# I will use the median imputation for replacing zero values.

fig, ax = plt.subplots(1,2, figsize=(20,6))
sns.distplot(ax=ax[0], x=data_prc.bedrooms,color='darkmagenta')
ax[0].set_title('Bedrooms', size=18)
sns.distplot(ax = ax[1], x = data_prc.bathrooms, color='darkmagenta')
ax[1].set_title('Bathrooms',size=18)


# In[87]:


data_prc.groupby('bedrooms')[['bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']].mean()


# In[89]:


# A great step in the data exploration is to dsiplay boxplot and outliers
sns.catplot(x='price',data=data_prc,kind='box',height=3,aspect=3)


# In[91]:


# Also we can print the probability distribution function (PDF) of a variable
# The PDF will show us how that variable is distributed 
# This makes it very easy to spot anomalies, icluded outliers
# The PDF is often the basis on which we decide whether we want to transform a feature
sns.distplot(data_prc.price, color='darkmagenta')


# In[92]:


# I will use the IQR measurement for removing outliers.
Q75=np.percentile(data_prc['price'],75)
Q25=np.percentile(data_prc['price'],25)
IQR=Q75-Q25
cutoff=IQR*1.5
upper=Q75 + cutoff
lower=1
data1=data_prc[(data_prc['price']<upper)]


# In[93]:


# what a change !
sns.distplot(data1.price,color='darkmagenta')
print(data_prc['price'].skew(),',',data1['price'].skew())


# In[95]:


data1.columns.values


# In[97]:


# lets look other variables
cols=['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']
fig, ax=plt.subplots(4,3, figsize=(30,15))
for idx, col in enumerate(cols):
    rn=math.floor(idx/3)
    cn=idx%3
    sns.distplot(ax=ax[rn,cn],x=data1[col],color='darkmagenta')
    ax[rn,cn].set_title(col, size=20)


# In[98]:


# bedrooms
print(data1.bedrooms.value_counts().sort_index())
sns.catplot(x='bedrooms',y='price',data=data1,height=3,aspect=3)


# In[100]:


data2=data1[data1.bedrooms<7]
# The number of houses with 7,8 and 9 bedrooms seems very low. 
# I will subtract these values
data2.bathrooms.value_counts().sort_index()


# In[102]:


data2.bathrooms=data2.bathrooms.astype(int)
print(data2.bathrooms.value_counts().sort_index())
sns.catplot(x='bathrooms',y='price',data=data2,height=3,aspect=3)


# In[103]:


data3=data2[data2.bathrooms<4]


# In[105]:


q = data3.sqft_living.quantile(0.99)
data4 = data3[data3.sqft_living<q]
print(data3.sqft_living.skew(),',', data4.sqft_living.skew())


# In[106]:


# There is still some skew
# But when we look at the price versus scatter plot, it looks much better now.

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10,3))

ax1.scatter(x= 'sqft_living', y= 'price', data = data3, c= 'sqft_living', cmap='inferno')
ax1.set_title('With Outliers')
ax1.set_xlim(0,7000)
ax1.set_ylim(-0.1e6,1.3e6)
ax2.scatter(x= 'sqft_living', y= 'price', data = data4, c= 'sqft_living', cmap='inferno')
ax2.set_title('Without Outliers')
ax2.set_xlim(0,7000)
ax2.set_ylim(-0.1e6,1.3e6)


# In[108]:


q = data4.sqft_lot.quantile(0.99)
data5 = data4[data4.sqft_lot<q]


# In[110]:


# let's compare the boxplots this time, there is a visible improvement.
fig, ax=plt.subplots(1,2, figsize=(10,6))
sns.boxplot(ax=ax[0], x=data4.sqft_lot, color='darkmagenta')
ax[0].set_title('With_outliers')
sns.boxplot(ax=ax[1], x=data5.sqft_lot, color='darkmagenta')
ax[1].set_title('Without_outliers')


# In[111]:


data5.floors.value_counts()


# In[114]:


# Again, I will convert data type to integer
data6=data5.copy()
data6.floors=data6.floors.astype(int)
print(data6.floors.value_counts())
sns.catplot(x='floors',y='price',data=data6,height=3,aspect=3)


# In[115]:


# As mentioned waterfall is divided in two categories, I will keep it way.
data6.waterfront.value_counts()


# In[118]:


#Since the number of house with a view is very few, it will be more useful for our analysis to see this feature as 0 and 1.
print(data6.view.value_counts())
data7=data6.copy()
data7.view=data7.view.map({0:0,1:1,2:1,3:1,4:1})
print(data7.view.value_counts())


# In[119]:


# There seems to be in bad shapes in USA,I'll remove 1's.
print(data7.condition.value_counts())
data7=data7[data7['condition']>1]


# In[122]:


# I will keep sqft_above as is.

data8=data7.copy()
sns.boxplot(data7.sqft_above,color='darkmagenta')

q = data8.sqft_basement.quantile(0.99)
data9 = data8[data8.sqft_basement<q]


# In[123]:


# There are many houses that do not have a basement.

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10,3))

ax1.scatter(x= 'sqft_basement', y= 'price', data = data8, c= 'sqft_basement', cmap='inferno')
ax1.set_title('With Outliers')
ax1.set_xlim(-100,2500)
ax1.set_ylim(-0.1e6,1.25e6)
ax2.scatter(x= 'sqft_basement', y= 'price', data = data9, c= 'sqft_basement', cmap='inferno')
ax2.set_title('Without Outliers')
ax2.set_xlim(-100,2500)
ax2.set_ylim(-0.1e6,1.25e6)


# In[124]:


# I will keep biulidng yeas as is
sns.boxplot(data9.yr_built, color='darkmagenta')


# In[125]:


# This is the last feature I will take as renovated(1) not (0)
sns.distplot(data9.yr_renovated, color='darkmagenta')


# In[126]:


data9.yr_renovated=pd.np.where(data9.yr_renovated==0,0,1)


# In[127]:


# We've removed most of the outliers. First I'll just continue with numeric values.
# Let's continue by dropping the categorical variables and saving it as a separate data set.
data_pp=data9.drop(['date', 'city', 'street', 'statezip', 'country'], axis=1)
data_pp.describe().T


# In[128]:


data_pp = data_pp.reset_index(drop=True)


# In[129]:


# Define the targets and inputs.

targets = data_pp.iloc[:,:1]
unscaled_inputs = data_pp.drop(['price'], axis = 1)

unscaled_inputs.head()


# In[130]:


# We will divide our data into train and test groups. This is important to avoid overfitting or underfitting.
# Overfitting means, our training has focused on the particular training data set so much, so it has missed the point. 
# Underfitting means the model has not captured the underlying logic of the data.

x_train, x_test, y_train, y_test = train_test_split(unscaled_inputs, targets, test_size=0.2, random_state=42)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# In[131]:


# I will keep features that contain only 0 or 1 data separately.

columns_to_scale = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'condition', 'sqft_above', 'sqft_basement',
       'yr_built']
columns_to_omit = ['waterfront','view','yr_renovated']
x_train_to_scale = x_train[columns_to_scale]
x_test_to_scale = x_test[columns_to_scale]
x_train_not_to_scale = x_train[columns_to_omit].reset_index(drop=True)
x_test_not_to_scale = x_test[columns_to_omit].reset_index(drop=True)


# In[132]:


# Let's fit the scaler using only the training data, then transform both training and test data.

scaler = RobustScaler()  #StandardScaler()
scaler.fit(x_train_to_scale)


# In[133]:


x_train_scaled = scaler.transform(x_train_to_scale)
x_test_scaled = scaler.transform(x_test_to_scale)
# After scaling our training and test data are converted to np.arrays,
# Let's make them pd.DataFrame again and merge them with unscaled features.

x_train_scaled = pd.DataFrame(x_train_scaled, columns=columns_to_scale)
x_train = pd.concat([x_train_scaled,x_train_not_to_scale], axis=1)
y_train = y_train.reset_index(drop=True)

x_test_scaled = pd.DataFrame(x_test_scaled, columns=columns_to_scale)
x_test = pd.concat([x_test_scaled,x_test_not_to_scale], axis=1)
y_test = y_test.reset_index(drop=True)


# In[134]:


# Now, we are ready!

x_train.head(2)


# In[135]:


# Let's create the model as we did at the beginning.

x = sm.add_constant(x_train)
results = sm.OLS(y_train, x).fit()

results.summary()


# In[136]:


# We talked about the low contribution of variables with P values above 0.05 to the model, 
# so I'm going to drop the renovation year value and run the model again.

x_train_pv = x_train.drop(['yr_renovated'], axis=1)
# R2 and Adj. R2 is the same as before. 
# We did well by removing it out, because it's always better to keep the equation simple.

X = sm.add_constant(x_train_pv)
results = sm.OLS(y_train, X).fit()

results.summary()


# In[137]:


data_wd = data9.drop(['date', 'street', 'statezip', 'country'], axis=1)
# Get_dummies is one of the common ways to create dummy variables for categorical features

city_dummies = pd.get_dummies(data_wd.city, drop_first = True)
city_dummies


# In[138]:


# Let's define the variables one more time.

targets = data_wd.iloc[:,:1]

a = data_wd.drop(['price','city'], axis = 1)
unscaled_inputs_wd = pd.concat([a, city_dummies], axis=1)
unscaled_inputs_wd.head(2)


# In[139]:


# Split the targets and inputs into train-test data again.

x_train, x_test, y_train, y_test = train_test_split(unscaled_inputs_wd, targets, test_size=0.2, random_state = 42)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[140]:


x_train.columns.values


# In[141]:


columns_to_scale = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'condition', 'sqft_above', 'sqft_basement',
       'yr_built']
columns_to_omit = ['waterfront','view','yr_renovated', 'Auburn', 'Beaux Arts Village',
       'Bellevue', 'Black Diamond', 'Bothell', 'Burien', 'Carnation',
       'Clyde Hill', 'Covington', 'Des Moines', 'Duvall', 'Enumclaw',
       'Fall City', 'Federal Way', 'Inglewood-Finn Hill', 'Issaquah',
       'Kenmore', 'Kent', 'Kirkland', 'Lake Forest Park', 'Maple Valley',
       'Medina', 'Mercer Island', 'Milton', 'Newcastle', 'Normandy Park',
       'North Bend', 'Pacific', 'Preston', 'Ravensdale', 'Redmond',
       'Renton', 'Sammamish', 'SeaTac', 'Seattle', 'Shoreline',
       'Skykomish', 'Snoqualmie', 'Snoqualmie Pass', 'Tukwila', 'Vashon',
       'Woodinville', 'Yarrow Point']
x_train_to_scale = x_train[columns_to_scale]
x_test_to_scale = x_test[columns_to_scale]
x_train_not_to_scale = x_train[columns_to_omit].reset_index(drop=True)
x_test_not_to_scale = x_test[columns_to_omit].reset_index(drop=True)


# In[142]:


# Scale as before

scaler = RobustScaler()
scaler.fit(x_train_to_scale)


# In[143]:


x_train_scaled = scaler.transform(x_train_to_scale)
x_test_scaled = scaler.transform(x_test_to_scale)
x_train_scaled = pd.DataFrame(x_train_scaled, columns=columns_to_scale)
x_train = pd.concat([x_train_scaled,x_train_not_to_scale], axis=1)
y_train = y_train.reset_index(drop=True)

x_test_scaled = pd.DataFrame(x_test_scaled, columns=columns_to_scale)
x_test = pd.concat([x_test_scaled,x_test_not_to_scale], axis=1)
y_test = y_test.reset_index(drop=True)


# In[144]:


# Yey! We managed to get much better results now.
# 'sqft_lot' and 'yr_renovated' features seems insignificant when we evaluate them according to their p-values.

X1 = sm.add_constant(x_train)
results = sm.OLS(y_train, X1).fit()

results.summary()


# In[147]:


x_train.columns.values


# In[148]:


# Let's check if there are multicollunearity.

from statsmodels.stats.outliers_influence import variance_inflation_factor

# all features where we want to check for multicollinearity:
variables = x_train[[ 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'condition',  'sqft_basement',
       'yr_built', 'view', 'yr_renovated', 'Auburn', 'Beaux Arts Village',
       'Bellevue', 'Black Diamond', 'Bothell', 'Burien', 'Carnation',
       'Clyde Hill', 'Covington', 'Des Moines', 'Duvall', 'Enumclaw',
       'Fall City', 'Federal Way', 'Inglewood-Finn Hill', 'Issaquah',
       'Kenmore', 'Kent', 'Kirkland', 'Lake Forest Park', 'Maple Valley',
       'Medina', 'Mercer Island', 'Milton', 'Newcastle', 'Normandy Park',
       'North Bend', 'Pacific', 'Preston', 'Ravensdale', 'Redmond',
       'Renton', 'Sammamish', 'SeaTac', 'Seattle', 'Shoreline',
       'Skykomish', 'Snoqualmie', 'Snoqualmie Pass', 'Tukwila', 'Vashon',
       'Woodinville', 'Yarrow Point']]

# we create a new data frame which will include all the VIFs, 
# each variable has its own VIF as this measure is variable specific (not model specific)
vif = pd.DataFrame()

# here we make use of the variance_inflation_factor, which will basically output the respective VIFs 
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]

# Finally, I will include names so it is easier to explore the result
vif["Features"] = variables.columns

vif.sort_values(by='VIF')


# In[149]:


# I will drop 'sqft_above' based on features VIF scores, 
# also 'sqft_lot' and 'yr_renovated' based on p-value that we have already determined.

x_train = x_train.drop(['sqft_above','sqft_lot','yr_renovated'], axis=1)
x_test = x_test.drop(['sqft_above','sqft_lot','yr_renovated'], axis=1)
# This time let's use sklearn to build our model.

reg = LinearRegression()
reg.fit(x_train, y_train)


# In[150]:


# Let's check the assumption of normality. 
# It seems quite good, right :)

y_hat = reg.predict(x_train)

sns.histplot(y_train - y_hat)


# In[151]:


# To measure adjusted R2, I will write a simple function. (Train)

def adj_R2(x_train,y_train):
  r2 = reg.score(x_train,y_train)
  n = x_train.shape[0]
  p = x_train.shape[1]
  adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
  return adjusted_r2

adj_R2 = adj_R2(x_train,y_train)
# To measure adjusted R2, I will write a simple function. (Test)

def adj_R2_test(x_test,y_test):
  r2 = reg.score(x_test,y_test)
  n = x_test.shape[0]
  p = x_test.shape[1]
  adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
  return adjusted_r2

adj_R2_test = adj_R2_test(x_test,y_test)


# In[152]:


# Our training and test scores are very similar,
# which means that overfitting was not observed in our model. 
print('R2-Train                      : {0:.2f}'.format(reg.score(x_train, y_train)*100))
print('R2-Test                       : {0:.2f}'.format(reg.score(x_test, y_test)*100))
print('Adj_R2-Train                  : {0:.2f}'.format(adj_R2*100))
print('Adj_R2-Test                   : {0:.2f}'.format(adj_R2_test*100))
print('MSE (Mean Squared Error)      : {0:.0f}'.format(metrics.mean_squared_error(y_train, y_hat)))
print('RMSE (Root Mean Squared Error): {0:.0f}'.format(np.sqrt(metrics.mean_squared_error(y_train, y_hat))))
print('MAE (Mean Ablosute Error)     : {0:.0f}'.format(metrics.mean_absolute_error(y_train, y_hat)))


# In[153]:


# Let's prepare a regression model summary table.

# Weights means for continuous variables: 
# positive weight = shows that as a feature increases in value, so do the price respectively
# negative weight = shows that as a feature increases in value, price decrease

# Weights means for dummy variables:
# positive weight = shows that the respective category(city) is more expensive than the benchmark
# negative weight = shows that the respective category(city) is less expensive than the benchmark 


reg_summary = pd.DataFrame(x_train.columns.values, columns = ['Features'])
reg_summary['Weights'] = reg.coef_.reshape(-1,1)
reg_summary.index = reg_summary.index +1
reg_summary.loc[0] = ['Intercept (b0)', reg.intercept_[0]]
reg_summary = reg_summary.sort_index()
reg_summary


# In[ ]:




