import numpy as np
from scipy.stats import invgauss
from scipy.stats import beta as beta_distr
from Scenario import Scenario


# Scenario 4 is a non-stationary scenario because the pdfs of the network parameters
# change continuously.
class Scenario4(Scenario):
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

	meanTH = 2.0327
	alphaLR = 0.075
	betaLR = 10
	meanDL = 0.637

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
		# Initialize network parameter pdf parameters for each step
		i = 0
		for x in np.arange(0, 9, 9/self.NSteps):
			self.funcTH[0, i] = np.exp(-x/3)
			self.funcTH[1, i] = -((x-4.5)**2)/150 + 0.5
			self.funcTH[2, i] = np.exp((x-9)/3)

			self.funcLR[0, i] = np.exp((x-9)/3)
			self.funcLR[1, i] = ((x-4.5)**2)/150 + 0.5
			self.funcLR[2, i] = np.exp(-x/3)

			self.funcDL[0, i] = np.exp((x-9)/3)
			self.funcDL[1, i] = ((x-4.5)**2)/150 + 0.5
			self.funcDL[2, i] = np.exp(-x/3)
			
			i += 1

	# Generate reward for arm at step n. If maxG = True, a reward from the optimal arm is obtained
	def generate_reward(self, arm, n, maxG = False):
		if maxG:
			arm = self.get_best_arm(n)

		th = self.getTH(arm, n)*invgauss.rvs(self.meanTH/5, scale = 5)*10
		lr = self.getLR(arm, n)*beta_distr.rvs(self.alphaLR, self.betaLR)
		dl = self.getDL(arm, n)*np.random.normal(self.meanDL, 0.1)

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
			QoE[arm] = self.getQoE(self.getTH(arm, n)*self.meanTH*10, self.getLR(arm, n)*(self.alphaLR/(self.alphaLR+self.betaLR)), self.getDL(arm, n)*self.meanDL)
		return np.argmax(QoE)

	# Obtain sample from TH pdf
	def getTH(self, arm, n):
		return self.funcTH[arm, n]

	# Obtain sample from DL pdf
	def getDL(self, arm, n):
		return self.funcDL[arm, n]

	# Obtain sample from LR pdf
	def getLR(self, arm, n):
		return self.funcLR[arm, n]

	
	