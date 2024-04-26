__author__ = 'lekesen'

# Import libraries & classes
import numpy as np
from scipy.stats import invgauss
from scipy.stats import beta as beta_distr
from Scenarios.Scenario import Scenario

# This scenario is a non-stationary contextual scenario because the pdfs of the network
# parameters change every 30k steps in discrete jumps. The context also keeps on changing...

class Scenario3Contextual(Scenario):
	NRuns = 3
	NSteps = 90000

	# Number of arms
	m = 5

	# Define the nm and nu values
	nm = 1
	nu = 5

	# Define values & parameters for TH, LR and DL
	funcTH = np.zeros((5, NSteps))
	funcDL = np.zeros((5, NSteps))
	funcLR = np.zeros((5, NSteps))
	
	"""
	meanTH = {
		'Zone1': [5, 1, 1, 1, 1],
		'Zone2': [1, 5, 1, 1, 1],
		'Zone3': [1, 1, 5, 1, 1]
	}
	gammaTH = {
		'Zone1': [25, 1, 1, 1, 1],
		'Zone2': [1, 25, 1, 1, 1],
		'Zone3': [1, 1, 25, 1, 1]
	}

	meanDL = {
		'Zone1': [0.05, 0.2, 0.2, 0.2, 0.2],
		'Zone2': [0.2, 0.05, 0.2, 0.2, 0.2],
		'Zone3': [0.2, 0.2, 0.05, 0.2, 0.2]
	}
	varDL = {
		'Zone1': [0.05, 0.05, 0.05, 0.05, 0.05],
		'Zone2': [0.05, 0.05, 0.05, 0.05, 0.05],
		'Zone3': [0.05, 0.05, 0.05, 0.05, 0.05]
	}

	meanLR = {
		'Zone1': [0.5, 5, 5, 5, 5],
		'Zone2': [5, 0.5, 5, 5, 5],
		'Zone3': [5, 5, 0.5, 5, 5]
	}
	alphaLR = 0.075
	betaLR = 10
	"""
	# Define values & parameters for TH, LR and DL

	meanTH = {
		'Zone1': [5, 2, 0.5, 0.5, 0.5],
		'Zone2': [3, 5, 3, 0.5, 0.5],
		'Zone3': [0.5, 2, 5, 0.5, 0.5]
	}
	gammaTH = {
		'Zone1': [25, 5, 1, 1, 1],
		'Zone2': [10, 50, 10, 1, 1],
		'Zone3': [1, 5, 25, 1, 1]

	}

	meanDL = {
		'Zone1': [0.05, 0.6, 1, 1, 1],
		'Zone2': [0.6, 0.4, 0.6, 1, 1],
		'Zone3': [1, 0.6, 0.05, 1, 1]
	}
	varDL = {
		'Zone1': [0.01, 0.1, 0.2, 0.1, 0.1],
		'Zone2': [0.1, 0.05, 0.1, 0.1, 0.1],
		'Zone3': [0.2, 0.1, 0.01, 0.1, 0.1]
	}

	meanLR = {
		'Zone1': [0.5, 1, 1.5, 2, 2],
		'Zone2': [1, 0.75, 1, 2, 2],
		'Zone3': [1.5, 1, 0.5, 2, 2]
	}
	alphaLR = 0.075
	betaLR = 10

	#'Web Browsing', 'Instant Message', 'Voice Call', 'Online Game', 'Meeting Video', 'On-Demand Audio', 'File Sharing', 'Location', 

	traffic_types = ['Live Audio', 'On-Demand Video', 'Live Video']

	def __init__(self):
		super().__init__()

	# Generate reward for arm atp step n. If maxG = True, a reward from the optimal arm is obtained
	def generate_reward(self, arm, n, maxG = False):
		if maxG:
			arm = self.get_best_arm(n)
			th = self.getTH(arm, n)
			lr = self.getLR(arm, n)
			dl = self.getDL(arm, n)
			return self.getQoE(th, lr, dl, n)

		th = self.getTH(arm, n)
		lr = self.getLR(arm, n)
		dl = self.getDL(arm, n)

		return self.getQoE(th, lr, dl, n)

	# Get QoE for the traffic type
	def getQoE(self, TH, LR, DL, n):
		context = self.getContextInternal(n)
		QoS = context[3]*np.exp(context[4]*DL) + context[5]/(LR+context[6]) + context[7]*np.log(TH) + context[8]
		QoE = context[0]*np.exp(-context[1]*QoS) + context[2]
		return QoE

	def getContext(self, n):
		index = int(n/5)%3
		return self.traffic_type_values[self.traffic_types[index]][3:9]# self.traffic_type_values['Live Video']

	def getContextInternal(self, n):
		index = int(n/5)%3
		return self.traffic_type_values[self.traffic_types[index]] #self.traffic_type_values['Live Video']

	# Get optimal arm at step n
	def get_best_arm(self, n):
		QoE = np.zeros(self.m)
		for arm in range(self.m):
			QoE[arm] = self.getQoE(self.getTH(arm, n, isMean = True), self.getLR(arm, n, isMean = True), self.getDL(arm, n, isMean = True), n)
		return np.argmax(QoE)

	# Obtain sample from TH pdf
	def getTH(self, arm, n, isMean = False):
		zone = 'Zone2'
		if n < 30000:
			zone = 'Zone1'
		elif n > 60000:
			zone = 'Zone3'

		if isMean:
			return (self.meanTH[zone])[arm]*10
		return invgauss.rvs(((self.meanTH[zone])[arm])/(self.gammaTH[zone])[arm], scale = (self.gammaTH[zone])[arm])*10

	# Obtain sample from DL pdf
	def getDL(self, arm, n, isMean = False):
		zone = 'Zone2'
		if n < 30000:
			zone = 'Zone1'
		elif n > 60000:
			zone = 'Zone3'

		if isMean:
			return (self.meanDL[zone])[arm]
		return max(np.random.normal((self.meanDL[zone])[arm], (self.varDL[zone])[arm]), 0)

	# Obtain sample from LR pdf
	def getLR(self, arm, n, isMean = False):
		zone = 'Zone2'
		if n < 30000:
			zone = 'Zone1'
		elif n > 60000:
			zone = 'Zone3'

		if isMean:
			return (self.meanLR[zone])[arm]*(self.alphaLR/(self.alphaLR+self.betaLR))
		return (self.meanLR[zone])[arm]*beta_distr.rvs(self.alphaLR, self.betaLR)
