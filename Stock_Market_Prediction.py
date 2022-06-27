#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[57]:


### Data collection
import pandas_datareader as pdr


# In[58]:


key="ce7428d2c9cce0ecf03f5bebd870f5e45f16858e"


# In[59]:


import pandas_dataframe as pdr
df=pdr.get_data_tiingo('AAPL',api_key=key)


# In[60]:


df.to_csv('AAPL.csv')


# In[61]:


import pandas as pd


# In[62]:


df=pd.read_csv('AAPL.csv')


# In[63]:


df.head()


# In[64]:


df.tail()


# In[65]:


df2=df.reset_index()['close']


# In[66]:


df2[1228:]


# In[67]:


df1=df.reset_index()['close']


# In[68]:


df1.shape


# In[69]:


import matplotlib.pyplot as plt
plt.plot(df1)


# In[70]:


### LSTM are sensitive to the scale of the data. so we can apply minmax scaler


# In[71]:


import numpy as np


# In[72]:


df1


# In[73]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[74]:


df1.shape


# In[75]:


df1.shape


# In[76]:


# splitting data into train test


# In[77]:


training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[78]:


training_size,test_size


# In[79]:


len(train_data),len(test_data)


# In[80]:


import numpy as np
# convert an array of value into dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY =[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step, 0])
    return np.array(dataX), np.array(dataY)


# In[81]:


# reshape into x=t, t+1,t+3 and y=t+4
time_step = 100
X_train,y_train=create_dataset(train_data,time_step)
X_test, y_test= create_dataset(test_data, time_step)


# In[82]:


print(X_train)


# In[83]:


# reshape input to be [samples, time steps, features] which is required in LSTM
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1], 1)
X_test =X_test.reshape(X_test.shape[0],X_test.shape[-1],1)


# In[84]:


# create LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[85]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1))) # LSTM hidden layer 50,input_shape=(100,1)
model.add(LSTM(50,return_sequences=True)) # Stakced LSTM 
model.add(LSTM(50))
model.add(Dense(1)) # Addind one final output
model.compile(loss='mean_squared_error', optimizer='adam') # Compling with mean square error and optimizer is adam 


# In[86]:


model.summary()


# In[87]:


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1)


# In[88]:


import tensorflow as tf


# In[89]:


tf.__version__


# In[90]:


# Lets do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[91]:


# Transform to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[92]:


### calculate RSME performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict)) # train predict is output for train data set


# In[93]:


# test data RSME
math.sqrt(mean_squared_error(y_test,test_predict))


# In[94]:


### Plotting
# shift train predictions for plotting
look_back=100 # consider time-step 100
trainPredictPlot=np.empty_like(df1)
trainPredictPlot[:,:]=np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :]=train_predict
# shift test predictions for plotting
testPredictPlot=np.empty_like(df1)
testPredictPlot[:,:]=np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1,:] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[95]:


len(test_data)


# In[106]:


x_input=test_data[341:].reshape(1,-1)
x_input.shape


# In[107]:


x_input=test_data[341:].reshape(1,-1)


# In[108]:


x_input.shape


# In[109]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[110]:


temp_input


# In[119]:


# demonstarte prediction for next 30 days
from numpy import array


first_output=[]
n_steps=100
i=0
while(i<30):
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input{}".format(i,x_input))
        x_input=x_input.reshape(-1,1)
        x_input=x_input.reshape((1,n_steps,1))
        # print(x_input)
        yhat=model.predict(x_input,verbose=0)
        print('{} day output{}'.format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        # print(temp_input)
        first_output.extend(yhat.tolist())
        i=i+1
        
    else:
        x_input=x_input.reshape((1, n_steps,1))
        yhat=model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        first_output.extend(yhat.tolist())
        i=i+1


# In[121]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[122]:


import matplotlib.pyplot as plt


# In[123]:


len(df1)


# In[124]:


df3=df1.tolist()


# In[125]:


df3.extend(first_output)


# In[127]:


plt.plot(day_new,scaler.inverse_transform(df1[1159:]))
plt.plot(day_pred,scaler.inverse_transform(first_output))


# In[131]:


# For complete output
df3=df1.tolist()
df3.extend(first_output)
plt.plot(df3[1000:])


# In[ ]:




