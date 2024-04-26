__author__ = 'lekesen'

# Import libraries & classes
import numpy as np
from scipy.stats import invgauss
from scipy.stats import beta as beta_distr
from Scenarios.Scenario import Scenario

# Scenario 2 is a non-stationary scenario. Even though the pdfs of the network parameters
# are stationary, the traffic types change over time. This implies that the mapping functions
# (QoS->QoE) also change, making the rewards be non-stationary (in "discrete" jumps).

class Scenario2Hard(Scenario):
	# TODO: Depending on environment variable have a value or another
	NRuns = 20 # 100
	NSteps = 100000

	# Number of arms
	m = 5

	# Define the nm and nu values
	# TODO: find best & more realistic values
	nm = 1
	nu = 5

	# Define parameters for the network parameter pdfs

	# Throughput --> Inverse Gaussian
	meanTH = [5, 4, 2, 1, 1]
	gammaTH = [10, 8, 4, 2, 2]

	# Delay --> Gaussian distribution
	meanDL = [0.75, 0.5, 0.25, 1, 1]
	varDL = [0.15, 0.1, 0.05, 0.2, 0.2]

	# Loss Rate --> Beta distribution
	meanLR = [0.75, 0.3, 1, 1.5, 1.5]
	alphaLR = 0.075
	betaLR = 10

	# Traffic type constants for the mapping functions
	# Constant traffic type change: Web Browsing, Instant Message,Voice Call, Online Game, Meeting Video
	traffic_types = ['Web Browsing', 'Instant Message', 'Voice Call', 'Online Game', 'Meeting Video', 'On-Demand Audio', 'File Sharing', 'Location', 'Live Audio', 'On-Demand Video', 'Live Video']

	def __init__(self):
		super().__init__()

	# Generate reward for arm at step n. If maxG = True, a reward from the optimal arm is obtained
	def generate_reward(self, arm, n, maxG = False):
		if maxG:
			arm = self.get_best_arm(n)

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

	# Get context (for Deep Thompson Sampling algorithm)
	def getContext(self, n):
		index = int(n/5)%10
		return (self.traffic_type_values[self.traffic_types[index]])[3:9]

	# Get context Internal
	def getContextInternal(self, n):
		index = int(n/5)%10
		return self.traffic_type_values[self.traffic_types[index]]

	# Get optimal arm at step n
	def get_best_arm(self, n):
		QoE = np.zeros(self.m)
		for arm in range(self.m):
			QoE[arm] = self.getQoE(self.getTH(arm, n, isMean = True), self.getLR(arm, n, isMean = True), self.getDL(arm, n, isMean = True), n)
		return np.argmax(QoE)

	# Obtain sample from TH pdf
	def getTH(self, arm, n, isMean = False):
		if isMean:
			return self.meanTH[arm]*10
		return invgauss.rvs((self.meanTH[arm])/self.gammaTH[arm], scale = self.gammaTH[arm])*10

	# Obtain sample from DL pdf
	def getDL(self, arm, n, isMean = False):
		if isMean:
			return self.meanDL[arm]
		return max(np.random.normal(self.meanDL[arm], self.varDL[arm]), 0)

	# Obtain sample from LR pdf
	def getLR(self, arm, n, isMean = False):
		if isMean:
			return self.meanLR[arm]*(self.alphaLR/(self.alphaLR+self.betaLR))
		return self.meanLR[arm]*beta_distr.rvs(self.alphaLR, self.betaLR)