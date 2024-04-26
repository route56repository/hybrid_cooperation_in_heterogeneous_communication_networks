from algorithms.auxiliar.ConcreteDropout import ConcreteDropout
#from ConcreteDropout import ConcreteDropout
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

"""

NN with 2 hidden layers and 256 units each. ReLU activation function
Adam optimizer with learning rate = 0.001

"""

class ContextualNN2():
	# fit model

	def create_model(self, m, wd=0, dd=1e-5):
		losses = []
		self.m = m
		x_in = Input(shape=(9+self.m,))
		x, loss = ConcreteDropout(Dense(256, activation='relu'), weight_regularizer=wd, dropout_regularizer=dd, is_mc_dropout=False)(x_in)
		losses.append(loss)

		x, loss = ConcreteDropout(Dense(256, activation='relu'), weight_regularizer=wd, dropout_regularizer=dd, is_mc_dropout=False)(x)
		losses.append(loss)

		mean, loss = ConcreteDropout(Dense(1), weight_regularizer=wd, dropout_regularizer=dd, is_mc_dropout=False)(x)
		losses.append(loss)

		log_var, loss = ConcreteDropout(Dense(1), weight_regularizer=wd, dropout_regularizer=dd, is_mc_dropout=False)(x)
		losses.append(loss)

		x_out = concatenate([mean, log_var])

		self.model = Model(inputs=x_in, outputs=x_out)
		for loss in losses:
			self.model.add_loss(loss)

		def heteroscedastic_loss(true, pred):
			n_outputs = pred.shape[1] // 2
			mean = pred[:, :n_outputs]
			log_var = pred[:, n_outputs:]
			precision = tf.math.exp(-log_var)
			return tf.reduce_sum(precision * (true - mean) ** 2. + log_var, -1)

		def mse_loss(true, pred):
			n_outputs = pred.shape[1] // 2
			mean = pred[:, :n_outputs]
			return tf.reduce_mean((true - mean) ** 2, -1)

		opt = Adam(learning_rate=0.001)
		self.model.compile(optimizer=opt, loss=heteroscedastic_loss, metrics=[mse_loss])

		assert len(self.model.layers[1].trainable_weights) == 3 # kernel, bias and dropout prob
		assert len(self.model.losses) == 4 # a loss for each Concrete Dropout layer

		return self.model


	def train(self, observed_triplets, num_epochs=30):
		batch_size = len(observed_triplets)
		array_len = len(observed_triplets)
		X = np.zeros((array_len, 9+self.m)) # xa_pairs
		Y = np.zeros(array_len) # observed rewards
		for i in range(array_len):
			triplet = observed_triplets[i]
			X[i, 0] = (triplet[0])[0]
			X[i, 1] = (triplet[0])[1]
			X[i, 2] = (triplet[0])[2]
			X[i, 3] = (triplet[0])[3]
			X[i, 4] = (triplet[0])[4]
			X[i, 5] = (triplet[0])[5]
			X[i, 6] = (triplet[0])[6]
			X[i, 7] = (triplet[0])[7]
			X[i, 8] = (triplet[0])[8]
			X[i, 9+triplet[1]] = 1 # triplet[1] = arm (starts from zero). FOr categorical encoding, one input for each arm and activate the according signal
			Y[i] = triplet[2]

		self.model.fit(X, Y, epochs=num_epochs, batch_size=batch_size, verbose=0) #1

	def predict(self, x, arm):
		# x is the context. Traffic type encoded with ID.
		# arm is the arm/ network. Arm ID is the ID it has in MAB algorithm.
		predict_array = np.zeros((1, 9+self.m))
		predict_array[0, 0] = x[0]
		predict_array[0, 1] = x[1]
		predict_array[0, 2] = x[2]
		predict_array[0, 3] = x[3]
		predict_array[0, 4] = x[4]
		predict_array[0, 5] = x[5]
		predict_array[0, 6] = x[6]
		predict_array[0, 7] = x[7]
		predict_array[0, 8] = x[8]
		predict_array[0, 9+arm] = 1

		prediction = self.model(predict_array) # use __call__() --> faster
		mean = ((np.array(prediction))[0])[0]
		var = ((np.array(prediction))[0])[1]
		var = np.exp(-var)
		
		return mean, var # return a value from a gaussian distribution