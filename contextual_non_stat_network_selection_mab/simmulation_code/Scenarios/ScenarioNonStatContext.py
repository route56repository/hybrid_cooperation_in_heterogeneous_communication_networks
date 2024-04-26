__author__ = 'lekesen'

# Import libraries & classes
import numpy as np
import random

from Scenarios.Scenario import Scenario

# Scenario explanation

class ScenarioNonStatContext():
	NRuns = 5
	NSteps = 100000

	# Number of arms
	m = 5

	# Define the nm and nu values
	nm = 1
	nu = 5

	# Reward generation
	mean = []
	var = 0.1

	# Contexts
	context = [
		[1.0, 1.0, 1.0, 0.7, 0.7],
		[0.5, 0.5, 1.0, 1.3, 1.3],
		[1.0, 1.0, 1.0, 1.0, 1.0],
		[1.0, 1.0, 0.8, 0.8, 0.8],
		[0.5, 0.5, 0.5, 1.0, 1.0]
	]

	def __init__(self):
		super().__init__()
		# Initialize mean vectors

		mean1 = []
		mean1_1 = np.zeros(5)
		mean1_2 = np.zeros(5)
		mean1_3 = np.zeros(5)
		for i in range(5):
			mean1_1[i] = 0.5
			mean1_2[i] = 0.5*0.75
			mean1_3[i] = 0.5*0.5
			if i>2:
				mean1_1[i] = 1.2
				mean1_2[i] = 1.2*0.75
				mean1_3[i] = 1.2*0.5
			
		mean1.append(mean1_1)
		mean1.append(mean1_2)
		mean1.append(mean1_3)

		mean2 = []
		mean2_1 = np.zeros(5)
		mean2_2 = np.zeros(5)
		mean2_3 = np.zeros(5)
		for i in range(5):
			mean2_1[i] = 0.7
			mean2_2[i] = 0.7*0.75
			mean2_3[i] = 0.7*0.5
		mean2.append(mean2_1)
		mean2.append(mean2_2)
		mean2.append(mean2_3)

		mean3 = []
		mean3_1 = np.zeros(5)
		mean3_2 = np.zeros(5)
		mean3_3 = np.zeros(5)
		for i in range(5):
			mean3_1[i] = 0.5
			mean3_2[i] = 0.5
			mean3_3[i] = 0.5
		mean3.append(mean3_1)
		mean3.append(mean3_2)
		mean3.append(mean3_3)

		mean4 = []
		mean4_1 = np.zeros(5)
		mean4_2 = np.zeros(5)
		mean4_3 = np.zeros(5)
		for i in range(5):
			mean4_1[i] = 1.2*0.5
			mean4_2[i] = 1.2*0.75
			mean4_3[i] = 1.2
			if i>2:
				mean4_1[i] = 0.5*0.5
				mean4_2[i] = 0.5*0.75
				mean4_3[i] = 0.5
		mean4.append(mean4_1)
		mean4.append(mean4_2)
		mean4.append(mean4_3)

		mean5 = []
		mean5_1 = np.zeros(5)
		mean5_2 = np.zeros(5)
		mean5_3 = np.zeros(5)
		for i in range(5):
			mean5_1[i] = 0.6*0.5
			mean5_2[i] = 0.6*0.75
			mean5_3[i] = 0.7
		mean5.append(mean5_1)
		mean5.append(mean5_2)
		mean5.append(mean5_3)

		self.mean.append(mean1)
		self.mean.append(mean2)
		self.mean.append(mean3)
		self.mean.append(mean4)
		self.mean.append(mean5)




	# Generate reward for arm at step n. If maxG = True, a reward from the optimal arm is obtained
	def generate_reward(self, arm, n, maxG = False):
		if maxG:
			arm = self.get_best_arm(n)

		return self.getQoE(arm, n)

	def getQoE(self, arm, n, isMean = False):
		context = self.getContext(n)

		# Could change if we make polynomial
		context_vector = np.zeros(len(context))

		zone = 1
		if n < 30000:
			zone = 0
		elif n > 60000:
			zone = 2

		for i in range(len(context)):
			context_vector[i] = context[i]

		if isMean:
			QoE = np.dot(context_vector.transpose(), (self.mean[arm])[zone]) #COULD CHANGE
		else:
			QoE = np.dot(context_vector.transpose(), (self.mean[arm])[zone]) + np.random.normal(0, self.var)
		return QoE

	# Get context
	def getContext(self, n):
		index = int(n/5)%5
		return self.context[index]

	# Get optimal arm at step n
	def get_best_arm(self, n):
		QoE = np.zeros(self.m)
		for arm in range(self.m):
			QoE[arm] = self.getQoE(arm, n, isMean = True)
		k = np.argmax(QoE)
		return k