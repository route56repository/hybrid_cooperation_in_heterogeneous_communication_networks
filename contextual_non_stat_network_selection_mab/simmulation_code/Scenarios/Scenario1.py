__author__ = 'lekesen'

# Import libraries & classes
import numpy as np
from scipy.stats import invgauss
from scipy.stats import beta as beta_distr
from Scenarios.Scenario import Scenario

# Scenario 1 is a stationary scenario. This means that Live Video is the only traffic type considered.
# It also implies that the quality of the networks (the fdps that generate the network parameter values
# are stationary).

class Scenario1(Scenario):
	# TODO: Depending on environment variable have a value or another
	NRuns = 5#100
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
	meanDL = [0.4, 0.6, 1, 0.5, 0.6]
	varDL = [0.05, 0.1, 0.2, 0.05, 0.1]

	# Loss Rate --> Beta distribution
	meanLR = [0.5, 1, 0.75, 0.6, 1]
	alphaLR = 0.075
	betaLR = 10

	# Traffic type constants for the mapping functions
	# 1 traffic type: Live Video
	a = 0.3965
	b = 0.8041
	c = -8.3946
	d = 0.2898
	p = -5.6192
	q = 0.3479
	alpha = 1.3566
	beta = 0.0424
	gamma = 1.5062


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
		QoS = self.a*np.exp(self.b*DL) + self.c/(LR+self.d) + self.p*np.log(TH) + self.q
		QoE = self.alpha*np.exp(-self.beta*QoS) + self.gamma
		return QoE

	# Get context (for Deep Thompson Sampling algorithm)
	def getContext(self, n):
		return self.traffic_type_values['Live Video']

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