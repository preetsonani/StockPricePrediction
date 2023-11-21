import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import pandas_datareader as data
from keras.models import load_model
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from numpy import array
import math
from sklearn.metrics import mean_squared_error
import numpy


st.title('Stock Price Prediction') 
user_input = st.text_input('Enter Stock Ticker', 'AAPL') # Getting user input for predicting stock price
 
data = yf.download(f'{user_input}', period = '5y') # downloading data from yahoo finance using yfinance library
data.to_csv(f'{user_input}.csv') # Data will be stored in data folder
df = pd.read_csv(f'{user_input}.csv')

date = pd.to_datetime(df.Date)

################# Describing Data ###################### will display data on user interface
#fig = plt.figure(figsize=(12,6))
#st.subheader('Data for the requested Stock')
#st.write(data.describe())
########################################################

# Visualizations of original graph, 100 days moving average and 200 days moving average using streamlit.
st.subheader('Closing Price vs Time Chart')
df1 = df.reset_index()['Close']
fig = plt.figure(figsize = (12,6))
plt.plot(date, df1)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
max100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(date, df.Close, label = 'Original graph')
plt.plot(date, max100, 'r', label = '100Days MA')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 & 200MA')
max100 = df.Close.rolling(100).mean()
max200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(date, df.Close, label = 'Original graph')
plt.plot(date, max100, 'r', label = '100Days MA')
plt.plot(date, max200, 'g', label = '200Days MA')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)


 # Scaling the data in range 0 to 1 using MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]    
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

########################################################## Create the Stacked LSTM model ######################################
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import LSTM

#model=Sequential()
#model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
#model.add(LSTM(50,return_sequences=True))
#model.add(LSTM(50))
#model.add(Dense(1))
#model.compile(loss='mean_squared_error',optimizer='adam')

#model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)
##################################################################################################################################
# THE ABOVE COMMENT IS THE STACKED LSTM MODEL WHICH WE TRAINED FOR STOCK PREDICTION. We saved the model as a external file so that it the model do not have to run each time.


# Load Model
model = load_model('keras_model.h5')

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Calculate RMSE performance metrics
math.sqrt(mean_squared_error(y_train,train_predict))

### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))

### Plotting 
st.subheader('Predicted graph')
fig = plt.figure(figsize=(12,6))


# shift train predictions for plotting
# shift train predictions for plotting
look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

# plot baseline and predictions
plt.plot(date, scaler.inverse_transform(df1), label = 'Original Graph')
plt.plot(date, trainPredictPlot, 'r', label = 'Train Prediction')
plt.plot(date, testPredictPlot, 'g', label = 'Test Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# Future Prediction of a stock price
x_input=test_data[341:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

lst_output=[]
n_steps=100
i=0
while(i<30):

    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i=i+1

day_new=np.arange(1,101)
day_pred=np.arange(101,131)

# Visualization of 10 days future prediction
st.subheader('Future Prediction for 10 Days')
fig = plt.figure(figsize=(12,6))
plt.plot(day_new,scaler.inverse_transform(df1[1158:]), label = ' Recent 20 days')
plt.plot(day_pred,scaler.inverse_transform(lst_output), label = '10 Days Future')
plt.xlabel('Volume')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)



