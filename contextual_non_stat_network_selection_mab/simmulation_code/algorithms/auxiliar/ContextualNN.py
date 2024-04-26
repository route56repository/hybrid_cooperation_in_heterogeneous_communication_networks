from algorithms.auxiliar.ConcreteDropout import ConcreteDropout
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
"""

NN with 2 hidden layers and 256 units each. ReLU activation function
Adam optimizer with learning rate = 0.001

"""
class ContextualNN():
	def __init__(self):
		# define the model
		self.model = Sequential()
		# input is an array of 2 elements. Output of 256 elements.
		self.model.add(ConcreteDropout(Dense(256, activation='relu'), input_shape=(2,)))
		self.model.add(ConcreteDropout(Dense(256, activation='relu')))
		self.model.add(Dense(1, activation='relu'))
		

		# compile the model
		opt = Adam(learning_rate=0.001)
		self.model.compile(optimizer=opt, loss=heteroscedastic_loss)

	def predict(self, x, arm):
		# x is the context. Traffic type encoded with ID.
		# arm is the arm/ network. Arm ID is the ID it has in MAB algorithm.
		predict_array = np.zeros((1,2))
		predict_array[0, 0] = x
		predict_array[0, 1] = arm
		
		
		return self.model(predict_array) # we are using __call__() (same as predict)

	def retrain(self, observed_triplets, epochs=50):
		array_len = len(observed_triplets)
		xa_pairs = np.zeros((array_len, 2))
		rewards = np.zeros(array_len)
		for i in range(array_len):
			triplet = observed_triplets[i]
			xa_pairs[i, 0] = triplet[0]
			xa_pairs[i, 1] = triplet[1]
			rewards[i] = triplet[2]
		
		self.model.fit(xa_pairs, rewards, epochs=epochs, batch_size=len(observed_triplets), verbose=0)