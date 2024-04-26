import numpy as np
from scipy.stats import invgauss
from scipy.stats import beta as beta_distr
from Scenario import Scenario

# Scenario 3 is a non-stationary scenario because the pdfs of the network parameters
# change every 30000 steps in discrete jumps.

class Scenario3(Scenario):
	NRuns = 100
	NSteps = 90000

	# Number of arms
	m = 3

	# Define the nm and nu values
	nm = 1
	nu = 5

	# Define values & parameters for TH, LR and DL
	funcTH = np.zeros((3, NSteps))
	funcLR = np.zeros((3, NSteps))
	funcDL = np.zeros((3, NSteps))

	meanTH = {
		'Zone1': [5, 2, 0.5],
		'Zone2': [3, 5, 3],
		'Zone3': [0.5, 2, 5]
	}
	gammaTH = {
		'Zone1': [25, 5, 1],
		'Zone2': [10, 50, 10],
		'Zone3': [1, 5, 25]

	}

	meanDL = {
		'Zone1': [0.05, 0.6, 1],
		'Zone2': [0.6, 0.4, 0.6],
		'Zone3': [1, 0.6, 0.05]
	}
	varDL = {
		'Zone1': [0.01, 0.1, 0.2],
		'Zone2': [0.1, 0.05, 0.1],
		'Zone3': [0.2, 0.1, 0.01]
	}

	meanLR = {
		'Zone1': [0.5, 1, 1.5],
		'Zone2': [1, 0.75, 1],
		'Zone3': [1.5, 1, 0.5]
	}
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
		

	# Generate reward for arm at step n. If maxG = True, a reward from the optimal arm is obtained
	def generate_reward(self, arm, n, maxG = False):
		if maxG:
			arm = self.get_best_arm(n)
			th = self.getTH(arm, n)
			lr = self.getLR(arm, n)
			dl = self.getDL(arm, n)
			return self.getQoE(th, lr, dl)

		th = self.getTH(arm, n)
		lr = self.getLR(arm, n)
		dl = self.getDL(arm, n)

		return self.getQoE(th, lr, dl)

	# Get QoE for the traffic type
	def getQoE(self, TH, LR, DL):
		QoS = self.a*np.exp(self.b*DL) + self.c/(LR+self.d) + self.p*np.log(TH) + self.q
		QoE = self.alpha*np.exp(-self.beta*QoS) + self.gamma
		return QoE

	# Get optimal arm at step n
	def get_best_arm(self, n):
		QoE = np.zeros(self.m)
		for arm in range(self.m):
			QoE[arm] = self.getQoE(self.getTH(arm, n, isMean = True), self.getLR(arm, n, isMean = True), self.getDL(arm, n, isMean = True))
		return np.argmax(QoE)

	# Obtain sample from TH pdf
	def getTH(self, arm, n, isMean = False):
		zone = "Zone2"
		if n < 30000:
			zone = "Zone1"
		elif n > 60000:
			zone = "Zone3" 

		if isMean:
			return (self.meanTH[zone])[arm]*10
		return invgauss.rvs(((self.meanTH[zone])[arm])/(self.gammaTH[zone])[arm], scale = (self.gammaTH[zone])[arm])*10

	# Obtain sample from DL pdf
	def getDL(self, arm, n, isMean = False):
		zone = "Zone2"
		if n < 30000:
			zone = "Zone1"
		elif n > 60000:
			zone = "Zone3" 
		
		if isMean:
			return (self.meanDL[zone])[arm]
		return np.random.normal((self.meanDL[zone])[arm], (self.varDL[zone])[arm])

	# Obtain sample from LR pdf
	def getLR(self, arm, n, isMean = False):
		zone = "Zone2"
		if n < 30000:
			zone = "Zone1"
		elif n > 60000:
			zone = "Zone3" 
		
		if isMean:
			return (self.meanLR[zone])[arm]*(self.alphaLR/(self.alphaLR+self.betaLR))
		return (self.meanLR[zone])[arm]*beta_distr.rvs(self.alphaLR, self.betaLR)
	