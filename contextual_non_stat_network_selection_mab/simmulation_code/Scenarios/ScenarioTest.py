__author__ = 'lekesen'

# Import libraries & classes
import numpy as np
from Scenarios.Scenario import Scenario

# Scenario 1 is a stationary scenario. This means that Live Video is the only traffic type considered.
# It also implies that the quality of the networks (the fdps that generate the network parameter values
# are stationary).

class ScenarioTest(Scenario):
	# TODO: Depending on environment variable have a value or another
	NRuns = 10#100
	NSteps = 100000

	# Number of arms
	m = 3

	# Define the nm and nu values
	# TODO: find best & more realistic values
	nm = 1
	nu = 5

	# Define parameters for the network parameter pdfs
	# Throughput --> Gaussian distribution
	"""
	meanTH = [5, 2, 0.5, 2.5, 1]
	varTH = [0.05, 0.1, 0.2, 0.05, 0.1]

	# Delay --> Gaussian distribution
	meanDL = [0.4, 0.6, 1, 0.5, 0.6]
	varDL = [0.05, 0.1, 0.2, 0.05, 0.1]

	# Loss Rate --> Gaussian distribution
	meanLR = [0.3, 0.7, 0.6, 0.5, 0.7]
	varLR = [0.005, 0.01, 0.02, 0.005, 0.01]
	"""
	meanTH = [5, 3.5, 0.5, 4.5, 3] #[5, 2, 0.5, 2.5, 1]
	varTH = [0.05, 0.1, 0.2, 0.05, 0.1]

	# Delay --> Gaussian distribution
	meanDL = [0.4, 0.6, 1, 0.45, 0.6]
	varDL = [0.05, 0.1, 0.2, 0.05, 0.1]

	# Loss Rate --> Gaussian distribution
	meanLR = [0.3, 0.45, 0.6, 0.4, 0.7]
	varLR = [0.005, 0.01, 0.02, 0.005, 0.01]
	
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
		th = np.random.normal(self.meanTH[arm], self.varTH[arm]) * 10
		while (th < 0):
			th = np.random.normal(self.meanTH[arm], self.varTH[arm]) * 10
		return th

	# Obtain sample from DL pdf
	def getDL(self, arm, n, isMean = False):
		if isMean:
			return self.meanDL[arm]
		dl = np.random.normal(self.meanDL[arm], self.varDL[arm])
		while dl < 0:
			dl = np.random.normal(self.meanDL[arm], self.varDL[arm])
		return dl

	# Obtain sample from LR pdf
	def getLR(self, arm, n, isMean = False):
		if isMean:
			return self.meanLR[arm]
		lr = np.random.normal(self.meanLR[arm], self.varLR[arm])
		while lr<0 or lr>1:
			lr = np.random.normal(self.meanLR[arm], self.varLR[arm])
		return lr