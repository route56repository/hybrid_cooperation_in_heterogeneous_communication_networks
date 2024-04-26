__author__ = 'lekesen'

# Import libraries & classes
import numpy as np
from scipy.stats import invgauss
from scipy.stats import beta as beta_distr
from Scenarios.Scenario import Scenario

# Scenario 2 is a non-stationary scenario. Even though the pdfs of the network parameters
# are stationary, the traffic types change over time. This implies that the mapping functions
# (QoS->QoE) also change, making the rewards be non-stationary (in "discrete" jumps).

class Scenario2QoSContext(Scenario):
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
	meanTH = [5, 2, 0.5, 2.5, 1]
	gammaTH = [10, 4, 1, 8, 4]

	# Delay --> Gaussian distribution
	meanDL = [0.6, 0.6, 1, 0.5, 0.4]
	varDL = [0.1, 0.1, 0.2, 0.05, 0.05]

	# Loss Rate --> Beta distribution
	meanLR = [0.5, 1, 0.75, 0.6, 1]
	alphaLR = 0.075
	betaLR = 10

	# Traffic type constants for the mapping functions
	# Every 20k steps traffic type changes: Web Browsing, Instant Message,Voice Call, Online Game, Meeting Video
	a = [0.0136, 0.3321, 1.1351, 0.2743, 0.7221, 0.3321]
	b = [0.9295, 1.2422, 1.2, 0.9683, 1.1079, 1.2422]
	c = [-0.0420, -0.4628, -2.6529, -0.0212, -1.4156]
	d = [0.4487, 0.3522, 0.4672, -0.8657, 0.1628]
	p = [-0.0182, -0.0937, -0.4855, -0.0033, -1.9646]
	q = [-0.0703, -1.0115, 1.0115, 0.0893, 1.8747]
	alpha = [0.5003, 0.2375, 1.2947, 0.8871, 1.1468]
	beta = [13.0852, 1.2047, 0.1943, 1.9647, 0.2167]
	gamma = [-0.0488, 2.0718, 0.8373, 0.5365, 1.8257]
	traffic_types = ['Web Browsing', 'Instant Message', 'Voice Call', 'Online Game', 'Meeting Video']

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
		index = int(n/20000)
		QoS = self.a[index]*np.exp(self.b[index]*DL) + self.c[index]/(LR+self.d[index]) + self.p[index]*np.log(TH) + self.q[index]
		QoE = self.alpha[index]*np.exp(-self.beta[index]*QoS) + self.gamma[index]
		return QoE

	# Get context (for Deep Thompson Sampling algorithm)
	def getContext(self, n):
		index = int(n/20000)
		return (self.traffic_type_values[self.traffic_types[index]])[3:9]

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