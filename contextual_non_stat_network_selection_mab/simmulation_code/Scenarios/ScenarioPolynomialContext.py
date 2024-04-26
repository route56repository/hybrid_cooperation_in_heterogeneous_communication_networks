__author__ = 'lekesen'

# Import libraries & classes
import numpy as np
import random
from scipy.stats import invgauss
from scipy.stats import beta as beta_distr
from Scenarios.Scenario import Scenario

# Scenario 1 is a stationary scenario. This means that Live Video is the only traffic type considered.
# It also implies that the quality of the networks (the fdps that generate the network parameter values
# are stationary).

class ScenarioPolynomialContext():
	# TODO: Depending on environment variable have a value or another
	NRuns = 5#100
	NSteps = 100000#100
	# Number of arms
	#m = 5
	m = 5

	# Define the nm and nu values
	# TODO: find best & more realistic values
	nm = 1
	nu = 5

	# Reward generation
	mean = []

	var = 0.1


	# Contexts
	context = [
		[0.5, 0.7, 0.6],
		[3.0, 2.0, 1.0],
		[1.0, 2.0, 0.5],
		[1.0, 2.0, 3.0],
		[0.7, 0.8, 0.1],
		[0.9, 4.0, 2.0]
	]
	#context = [
	#	[1.0, 2.0, 3.0, 4.0, 5.0],
	#	[3.0, 2.0, 4.0, 5.0, 6.0],
	#	[7.0, 4.0, 5.0, 1.0, 2.0],
	#	[3.0, 1.0, 4.0, 2.0, 6.0],
	#	[5.0, 6.0, 3.0, 4.0, 1.0],
	#	[4.0, 4.0, 4.0, 3.0, 3.0]
	#]

	def __init__(self):
		super().__init__()
		'''
		mean1 = np.zeros(5)
		for i in range(5):
			if i < 3:
				mean1[i] = 1
			else:
				mean1[i] = 0.7
		self.mean.append(mean1)

		mean2 = np.zeros(5)
		for i in range(5):
			mean2[i] = 0.7
		self.mean.append(mean2)

		mean3 = np.zeros(5)
		for i in range(5):
			if i > 2:
				mean3[i] = 1
			else:
				mean3[i] = 0.7
		self.mean.append(mean3)

		mean4 = np.zeros(5)
		for i in range(5):
			mean4[i] = 0.5
		self.mean.append(mean4)

		mean5 = np.zeros(5)
		for i in range(5):
			mean5[i] = 0.3
		self.mean.append(mean5)
		'''
		mean1 = np.zeros(10)
		for i in range(10):
			if i < 6:
				mean1[i] = 1
			else:
				mean1[i] = 0.8
		self.mean.append(mean1)

		mean2 = np.zeros(10)
		for i in range(10):
			mean2[i] = 0.8
		self.mean.append(mean2)

		mean3 = np.zeros(10)
		for i in range(10):
			mean3[i] = 0.6
		self.mean.append(mean3)

		mean4 = np.zeros(10)
		for i in range(10):
			if i >= 6:
				mean4[i] = 1
			else:
				mean4[i] = 0.8
		self.mean.append(mean4)

		mean5 = np.zeros(10)
		for i in range(10):
			mean5[i] = 0.3
		self.mean.append(mean5)



	# Generate reward for arm at step n. If maxG = True, a reward from the optimal arm is obtained
	def generate_reward(self, arm, n, maxG = False):
		if maxG:
			arm = self.get_best_arm(n)

		return self.getQoE(arm, n)


	def getQoE(self, arm, n, isMean=False):
		context = self.getContext(n)

		
		context_vector = np.ones(10)
		
		context_vector[0] = 1
		context_vector[1] = context[0]
		context_vector[2] = context[1]
		context_vector[3] = context[2]
		context_vector[4] = context[0]**2
		context_vector[5] = context[1]**2
		context_vector[6] = context[2]**2
		context_vector[7] = context[0]*context[1]
		context_vector[8] = context[0]*context[2]
		context_vector[9] = context[1]*context[2]
		"""

		context_vector = np.zeros(19)
		
		context_vector[0] = 1
		context_vector[1] = context[0]
		context_vector[2] = context[1]
		context_vector[3] = context[2]
		context_vector[4] = context[0]**2
		context_vector[5] = context[1]**2
		context_vector[6] = context[2]**2

		context_vector[7] = context[0]**3
		context_vector[8] = context[1]**3
		context_vector[9] = context[2]**3

		context_vector[10] = context[0]*context[1]
		context_vector[11] = context[0]*context[2]
		context_vector[12] = context[1]*context[2]

		context_vector[13] = (context[0]**2)*context[1]
		context_vector[14] = (context[0]**2)*context[2]

		context_vector[15] = (context[1]**2)*context[0]
		context_vector[16] = (context[1]**2)*context[2]

		context_vector[17] = (context[2]**2)*context[0]
		context_vector[18] = (context[2]**2)*context[1]
		"""
		if isMean:
			QoE = np.dot(context_vector.transpose(), self.mean[arm])
		else:
			QoE = np.dot(context_vector.transpose(), self.mean[arm]) + np.random.normal(0, self.var)
		return QoE

	# Get context
	def getContext(self, n):
		index = int(n/5)%6
		return self.context[index]

	# Get optimal arm at step n
	def get_best_arm(self, n):
		QoE = np.zeros(self.m)
		for arm in range(self.m):
			QoE[arm] = self.getQoE(arm, n, isMean=True)
			#print(QoE[arm])
		k = np.argmax(QoE)
		#print('Step ' + str(n) + '. Best arm: ' + str(k))
		return k
