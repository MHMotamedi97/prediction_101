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

x_train = tf.convert_to_tensor(np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1])))
x_test = tf.convert_to_tensor(np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1])))
y_train = tf.convert_to_tensor(y_train.reshape((y_train.size, 1)))
y_test = tf.convert_to_tensor(y_test.reshape((y_test.size, 1)))

model = Sequential()
model.add(LSTM(25, input_shape=(1, 10), return_sequences=True, activation = 'tanh', stateful=False))
# model.add(LSTM(50, return_sequences=True, activation = 'tanh', stateful=False))
# model.add(Dense(units=10, activation='relu'))

# model.add(LSTM(50, return_sequences=True, activation = 'tanh', stateful=False))
model.add(Dense(units=15, activation='relu'))
model.add(Dense(units=1)) 
model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True))
model.summary()