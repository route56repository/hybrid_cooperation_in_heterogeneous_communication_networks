__author__ = 'lekesen'

# Import libraries & classes
import numpy as np
import random

from Scenarios.Scenario import Scenario

# Scenario explanation

class ScenarioNonStatCubicContext():
	NRuns = 3
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
		[0.5, 0.7, 0.6],
		[3.0, 2.0, 1.0],
		[1.0, 2.0, 0.5],
		[1.0, 2.0, 3.0],
		[0.7, 0.8, 0.1],
		[0.9, 4.0, 2.0]
	]

	def __init__(self):
		super().__init__()
		# Initialize mean vectors
		raw_context_len = 3
		context_len = 0
        for pos in range(raw_context_len):
            context_len += pos
            context_len += raw_context_len - 1
        
        context_len += raw_context_len*3+1

		mean1 = []
		mean1_1 = np.zeros(context_len)
		mean1_2 = np.zeros(context_len)
		mean1_3 = np.zeros(context_len)
		for i in range(5):
			mean1_1[i] = 0.5
			mean1_2[i] = 0.5*0.75
			mean1_3[i] = 0.5*0.5
			if i>8:
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
			if i>8:
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
		raw_context_len = np.zeros(len(context))

		context_len = 0
        for pos in range(raw_context_len):
            context_len += pos
            context_len += raw_context_len - 1
        
        context_len += raw_context_len*3+1

		zone = 1
		if n < 30000:
			zone = 0
		elif n > 60000:
			zone = 2

		context_vector = np.zeros(context_len)

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

		if isMean:
			QoE = np.dot(context_vector.transpose(), (self.mean[arm])[zone]) #COULD CHANGE
		else:
			QoE = np.dot(context_vector.transpose(), (self.mean[arm])[zone]) + np.random.normal(0, self.var)
		return QoE

	# Get context
	def getContext(self, n):
		index = int(n/5)%6
		return self.context[index]

	# Get optimal arm at step n
	def get_best_arm(self, n):
		QoE = np.zeros(self.m)
		for arm in range(self.m):
			QoE[arm] = self.getQoE(arm, n, isMean = True)
		k = np.argmax(QoE)

		print("Step " + str(n))
		print("Arm " + str(k))

		return k