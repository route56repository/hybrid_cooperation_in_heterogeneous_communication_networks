__author__ = 'lekesen'

# Import libraries & classes
import numpy as np
from scipy.stats import invgauss
from scipy.stats import beta as beta_distr
from Scenarios.Scenario import Scenario

# Scenario 1 is a stationary scenario. This means that Live Video is the only traffic type considered.
# It also implies that the quality of the networks (the fdps that generate the network parameter values
# are stationary).

class ScenarioPeriodLinearContext():
	# TODO: Depending on environment variable have a value or another
	NRuns = 10
	NSteps = 100000

	# Number of arms
	m = 5

	# Define the nm and nu values
	# TODO: find best & more realistic values
	nm = 1
	nu = 5

	# Reward generation
	mean = []

	var = 0.2


	# Contexts
	context = [
		[1.0, 2.0, 3.0, 4.0, 5.0],
		[3.0, 2.0, 4.0, 5.0, 6.0],
		[7.0, 4.0, 5.0, 1.0, 2.0],
		[3.0, 1.0, 4.0, 2.0, 6.0],
		[5.0, 6.0, 3.0, 4.0, 1.0],
		[4.0, 4.0, 4.0, 3.0, 3.0]
	]

	def __init__(self):
		super().__init__()

		mean1 = np.zeros(5)
		for i in range(5):
			if i < 3:
				mean1[i] = 1
			else:
				mean1[i] = 0.8
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
				mean3[i] = 0.8
		self.mean.append(mean3)

		mean4 = np.zeros(5)
		for i in range(5):
			mean4[i] = 0.5
		self.mean.append(mean4)

		mean5 = np.zeros(5)
		for i in range(5):
			mean5[i] = 0.3
		self.mean.append(mean5)

	# Generate reward for arm at step n. If maxG = True, a reward from the optimal arm is obtained
	def generate_reward(self, arm, n, maxG = False):
		if maxG:
			arm = self.get_best_arm(n)

		return self.getQoE(arm, n)

	# MAYBE THIS SHOULD CHANGE!!!
	def getQoE(self, arm, n, isMean=False):
		context = self.getContext(n)

		context_vector = np.zeros(len(context))
		for i in range(len(context)):
			context_vector[i] = context[i]

		QoE = np.dot(context_vector, self.mean[arm]) + np.random.normal(0, self.var)
		
		if isMean:
			QoE = np.dot(context_vector.transpose(), self.mean[arm])

		return QoE

	# Get context
	def getContext(self, n):
		index = int(np.floor(n/20000))
		return self.context[index]

	# Get optimal arm at step n
	def get_best_arm(self, n):
		QoE = np.zeros(self.m)
		for arm in range(self.m):
			QoE[arm] = self.getQoE(arm, n, isMean=True)
		k = np.argmax(QoE)
		#print('Step ' + str(n) + '. Best arm: ' + str(k))
		return k