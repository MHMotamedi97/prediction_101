import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from binance.client import Client
import numpy as np
import pandas as pd
from datetime import datetime
import enum

data = pd.read_csv("bitcoin.csv")
data.info()

close = data.filter(items=['Close'])

close = close.values

scaler = MinMaxScaler()
scaler.fit(close)
close = scaler.transform(close)

train, test = train_test_split(close, test_size=0.2, random_state=42, shuffle=False)

look_back = 10

dataX, dataY = [], []
for i in range(len(train)-look_back-1):
    c = train[i:(i+look_back), 0]
    dataX.append(c)
    dataY.append(train[i + look_back, 0])
x_train = np.array(dataX)
y_train = np.array(dataY)

dataX, dataY = [], []
for i in range(len(test)-look_back-1):
    c = test[i:(i+look_back), 0]
    dataX.append(c)
    dataY.append(test[i + look_back, 0])
x_test = np.array(dataX)
y_test = np.array(dataY)

x_train = (np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1])))
x_test = (np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1])))
y_train = (y_train.reshape((y_train.size, 1)))
y_test = (y_test.reshape((y_test.size, 1)))

model = Sequential()
model.add(LSTM(25, input_shape=(1, look_back), return_sequences=True, activation = 'tanh', stateful=False))
# model.add(LSTM(50, return_sequences=True, activation = 'tanh', stateful=False))
# model.add(Dense(units=10, activation='relu'))

# model.add(LSTM(50, return_sequences=True, activation = 'tanh', stateful=False))
model.add(Dense(units=15, activation='relu'))
model.add(Dense(units=1)) 
model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True))
history = model.fit(x_train, y_train, epochs=100, batch_size=10)
print(model.evaluate(x=x_test, y=y_test, batch_size=10, verbose=1, sample_weight=None, steps=None, callbacks=None, max_queue_size=10, workers=8, use_multiprocessing=False))
model.save("bitcoin_model_close.h5")

