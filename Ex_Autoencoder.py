
# Import Libraries
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Input, Dense, GaussianNoise,Lambda,Dropout, concatenate, Layer
from keras.models import Model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,SGD
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pprint import pprint
import matplotlib.pyplot as plt
#matplotlib inline

# Set random seeds
from numpy.random import seed
seed(1)
tf.random.set_seed(3)

# Encoder Parameters
M = 16 # Nº de bits 128
N = 100000 # Amostras 100000
Columns = int(M/2)
Rows = int(N/Columns)
num_decimal = list()

#generating data of size N
label = np.random.randint(2,size=[N,M])#[rows, columns]
#print(label)

total = ""

for j in range(N):
	for element in label[j]:
		#turning line into string
		total += str(element)
	
	#print(total.encode())

	#turning string-bunary into decimal
	total_conv = int(total,2)
	#print(total_conv)
	
	#adding it to a list
	num_decimal.append(total_conv)
	total=""	

#num_decimal = np.array(num_decimal)

#turning list to tensor
num_dec = tf.reshape(num_decimal, [Rows,Columns])#[rows, columns]
#print(num_dec)	

x_train, x_test, y_train, y_test = train_test_split(num_dec,num_dec, test_size=0.2, random_state=42)
pprint(num_dec)
#print(label.ndim)

input_num = Input(shape = (M,))
encoded = Dense(256, activation='linear')(input_num)
encoded = Dense(256, activation='linear')(encoded)
encoded = Dense(256, activation='linear')(encoded)

decoded = Dense(256, activation='linear')(encoded)
decoded = Dense(256, activation='linear')(decoded)
decoded = Dense(128, activation='linear')(decoded)

autoencoder = Model(input_num, decoded)

#encoder = Model(input_num, encoded)
autoencoder.compile(optimizer = 'sgd', loss = 'mean_squared_error')

history  = autoencoder.fit(x_train,y_train, epochs = 100, 
				batch_size = Columns, 
				shuffle=True,
				validation_split=0.2)

plt.figure()
plt.plot(np.sqrt(history.history["loss"]))
plt.plot(np.sqrt(history.history["val_loss"]))
plt.title("Model loss")
plt.ylabel("sqrt(MSE)")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")

y_nn = autoencoder.predict(x_test)
