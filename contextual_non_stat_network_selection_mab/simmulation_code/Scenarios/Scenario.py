__author__ = 'lekesen'

# Scenario parent class

class Scenario:

	def __init__(self):
		# Define ID for each traffic type (for Deep Thompson Sampling algorithm)
		self.traffic_type_values = {
			#'On-Demand Audio': 1000,
			#'Web Browsing': 1001,
			#'Instant Message': 1002,
			#'File Sharing': 1003,
			#'Online Game': 1004,
			#'Live Video': 1005,
			#'Location': 1006,
			#'Live Audio': 1007,
			#'On-Demand Video': 1008,
			#'Meeting Video': 1009,
			#'Voice Call': 1010
			# alpha, beta, gamma, a, b, c, d, p, q
			'On-Demand Audio': [1.1466, 0.0299, -0.3209, 7.6178, 0.5831, -15.2617, 0.6199, -12.1927, -0.0103],
			'Web Browsing': [0.5003, 13.0852, -0.0488, 0.0136, 0.9295, -0.0420, 0.4487, -0.0182, -0.0703],
			'Instant Message': [0.2375, 1.2047, 2.0718, 0.3321, 1.2422, -0.4628, 0.3522, -0.0937, -1.0115],
			'File Sharing': [1.4757, 0.4038, 2.0749, 0.5913, 1.1134, -1.2989, 0.4902, -0.3592, 0.4521],
			'Online Game': [0.8771, 1.9647, 0.5635, 0.2743, 0.9683, -0.0212, -0.8657, -0.0033, 0.0893],
			'Live Video': [1.6466, 0.0293, 0.127, 4.3215, 0.7696, -14.1519, 0.4361, -8.6823, 0.0838],
			'Location': [1.0901, 0.2858, 1.5876, 0.7781, 1.048, -1.9215, 0.4574, -0.1893, 0.5047],
			'Live Audio': [1.1085, 0.7593, 1.5062, 0.3965, 1.0749, -0.8575, -0.7646, -0.0445, -2.7661],
			'On-Demand Video': [1.3566, 0.0424, -0.0199, 2.8019, 0.8041, -8.3946, 0.2898, -5.6192, 0.3479],
			'Meeting Video': [1.1468, 0.2167, 1.8257, 0.7221, 1.1079, -1.4156, 0.1628, -1.9646, 1.8747],
			'Voice Call': [1.2947, 0.1943, 0.8373, 1.1351, 1.2, -2.6529, 0.4672, -0.4855, 1.0115]
		}

	# Generate reward for arm at step n. If maxG = True, a reward from the optimal arm is obtained
	def generate_reward(self, arm, n, maxG = False):
		pass

	# Get optimal arm at step n
	def get_best_arm(self, n):
		pass

	# Get QoE for the traffic type
	def getQoE(self, TH, LR, DL):
		pass

	# Get context (for Deep Thompson Sampling algorithm)
	def getContext(self, n):
		pass