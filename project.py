import streamlit as st
import time
from tqdm.notebook import tqdm
from tensorflow import keras
import datetime as dt
from datetime import date
import yfinance as yf
import pandas as pd
import plotly.express as px, plotly.graph_objects as go
import imp
import plotly.figure_factory as ff

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

df = pd.read_csv("ITC.NS.csv")
data = df.dropna()
START = "2017-01-01"
TODAY = dt.datetime.now().strftime("%Y-%m-%d")


st.title('stock trend prediction')

user_input = st.text_input('Enter Stock Ticker', 'ITC.NS')

#describing data
st.subheader('Data from 2017-2022')

st.write(data.head())

st.subheader('Closing price vs Date')

fig = plt.figure(figsize = (12,6))
plt.plot(data.Close)
st.pyplot(fig)

st.subheader("closing price vs ma100 vs ma200")
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'b')

plt.plot(data.Close,'g')

plt.legend()

plt.show()
st.pyplot(fig)

cls = data[['Close']]
ds = cls.values

#normalizing the data
normalizer = MinMaxScaler(feature_range=(0,1))
ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))

#splittimg the data into training and testing data

train_size = int(len(ds_scaled)*0.70)
test_size = len(ds_scaled) - train_size
ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1]

#creating dataset in time series for LSTM model 
#X[100,120,140,160,180] : Y[200]
def create_ds(dataset,step):
    Xtrain, Ytrain = [], []
    for i in range(len(dataset)-step-1):
        a = dataset[i:(i+step), 0]
        Xtrain.append(a)
        Ytrain.append(dataset[i + step, 0])
    return np.array(Xtrain), np.array(Ytrain)

#Taking 100 days price as one record for training
time_stamp = 100
X_train, y_train = create_ds(ds_train,time_stamp)
X_test, y_test = create_ds(ds_test,time_stamp)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

#Creating LSTM model using keras
model = Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
model.add(LSTM(units=50,return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1,activation='linear'))
model.summary()

#Training model with adam optimizer and mean squared error loss function
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64)
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = normalizer.inverse_transform(train_predict)
test_predict = normalizer.inverse_transform(test_predict)
test = np.vstack((train_predict,test_predict))

st.subheader("original vs predicted")
fig1 = plt.figure(figsize = (12,6))

plt.plot(normalizer.inverse_transform(ds_scaled), 'g', label = 'original price')
plt.plot((test), 'b', label = 'predicted price')

plt.legend()

plt.show()
st.pyplot(fig1)

fut_inp = ds_test[271:]

fut_inp = fut_inp.reshape(1,-1)
tmp_inp = list(fut_inp)

#Creating list of the last 100 data
tmp_inp = tmp_inp[0].tolist()

#Predicting next 10 days price suing the current data
#It will predict in sliding window manner (algorithm) with stride 1
lst_output=[]
n_steps=100
i=0
while(i<10):
    
    if(len(tmp_inp)>100):
        fut_inp = np.array(tmp_inp[1:])
        fut_inp=fut_inp.reshape(1,-1)
        fut_inp = fut_inp.reshape((1, n_steps, 1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        tmp_inp = tmp_inp[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        fut_inp = fut_inp.reshape((1, n_steps,1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i=i+1
    
#Creating a dummy plane to plot graph one after another
ds_new = ds_scaled.tolist()


st.subheader("Forecasted Values")
forecast = normalizer.inverse_transform(lst_output)
st.write("forecast")
forecast








