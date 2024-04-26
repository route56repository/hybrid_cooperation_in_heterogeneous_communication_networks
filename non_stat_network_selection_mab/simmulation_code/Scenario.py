# Scenario parent class

class Scenario:
	def generate_reward(self, arm, n, maxG = False):
		pass

	def get_best_arm(self, n):
		pass

	def getQoE(self, TH, LR, DL):
		pass