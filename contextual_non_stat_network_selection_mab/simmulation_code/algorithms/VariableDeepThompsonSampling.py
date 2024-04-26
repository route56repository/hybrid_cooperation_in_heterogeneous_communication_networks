__author__ = 'lekesen'

# Import libraries & classes
from algorithms.auxiliar.ContextualNN_2 import ContextualNN2
import numpy as np
import time

'''
Deep Contextual MAB says...
In the below experiments for the contextual models (including the epsilon-greedy and fixed dropout rate bandits),
we use a NN with 2 hidden layers with 256 units each and ReLU activation function. All networks are trained using
the Adam optimizer with initial learning rate = 0.001. We retrain the contextual model on an exponential scale:
K = 2 and N = 128
'''

def VariableDeepThompsonSampling(scenario, maxG, verbose=True):
	# Get necessary parameters from the environment and simmulation
	NSteps = scenario.NSteps
	NRuns = scenario.NRuns
	m = scenario.m
	nu = scenario.nu
	nm = scenario.nm # check if we could use them

	# Parameters to save obtained results (regret, execution time & BA)
	avg_regret = np.zeros(NSteps)
	G = np.zeros((NRuns, NSteps))

	exec_time = np.zeros(NRuns)

	BA = np.zeros((NRuns, NSteps))
	avg_BA = np.zeros(NSteps)

	# Parameters for Deep Thompson Sampling
	# TODO: Put in scenario / other config file
	K = 1.05
	N = 15

	# Loop for each run
	for i in range(NRuns):
		# Initialize values
		n = 0 # initial step

		retrain_window = 2000 #3000
		retrain_window_step = 1000
		retrain_window_max = 10000
		retrain_window_min = 1000

		num_epochs = 20
		num_epochs_step = 5
		num_epochs_max = 30
		num_epochs_min = 5

		m0 = 10 #7
		m1 = 10 #7
		accumulated_m0 = 0
		accumulated_m1 = 0
		previous_sign = "-"

		next_retrain = N

		# Initialize neural network
		nn = ContextualNN2()
		nn.create_model(m)

		t0 = time.time()

		# Get initial measures to start training NN...
		initial_rewards = []
		for repetition in range(1):
			for arm in range(m):
				for step in range(nm):
					x = scenario.getContext(n)
					r = scenario.generate_reward(arm, n)
					initial_rewards.append([x, arm, r])
					n += 1
		nn.train(initial_rewards, num_epochs=50) # try to change epochs ?
		# Start main loop
		while n<NSteps:
			retrain_counter = 0
			observed_triplets = []
			while retrain_counter<next_retrain and n<NSteps:
				x = scenario.getContext(n)

				# TODO: sample dropout noise d
				# TODO: compute corresponding weights w_d
				# Compute predicted reward for each arm
				r = np.zeros(m)
				mean = np.zeros(m)
				var = np.zeros(m)
				#print("Step " + str(n))
				for arm in range(m):
					mean[arm], var[arm] = nn.predict(x, arm)
					r[arm] = np.random.normal(mean[arm], var[arm])

				#print("Estimations at step " + str(n), end=" ")
				#for arm in range(m):
				#	print("arm: " + str(arm) + " estimate: " + str(r[arm]))

				# Select arm with best predicted reward
				k = np.argmax(r)

				# Get rewards for selected arm (nu at the moment)
				j = 0
				g = 0
				while (j<nu) and ((j+n)<NSteps):
					w = scenario.generate_reward(k, n+j)
					g += w
					j += 1
					observed_triplets.append([x, k, w])
					if (n+j)%1000 == 0 and (n+j)!=NSteps:
						real_r = np.zeros(m)
						for arm in range(m):
							real_r[arm] = scenario.generate_reward(arm, n+j)
						print('Step ' + str(n+j) + '\t Selected arm: ' + str(k) + '\t Predicted reward: ' + str(r) + '\t Predicted mean: ' + str(mean) + '\t Predicted var: ' + str(var) + '\t Best arm: ' + str(scenario.get_best_arm(n)) +'\t Real reward: ' + str(real_r))
					
					sign = "+"
					if w < r[k]:
						sign = "-"

					if previous_sign == sign: # decrease memory
						accumulated_m1 += 1
						accumulated_m0 = 0

						if accumulated_m1 == m1:
							if retrain_window != retrain_window_min:
								retrain_window -= retrain_window_step
							if num_epochs != num_epochs_max:
								num_epochs += num_epochs_step
							accumulated_m1 = 0
					else:
						accumulated_m0 += 1
						accumulated_m1 = 0

						if accumulated_m0 == m0:
							if retrain_window != retrain_window_max:
								retrain_window += retrain_window_step
							if num_epochs != num_epochs_min:
								num_epochs -= num_epochs_step
							accumulated_m0 = 0

					previous_sign = sign

				# Save observed rewards & info
				if k == scenario.get_best_arm(n):
					BA[i, n:(n+j)] = 1

				G[i, n:(n+j)] = g + G[i, n-1]

				retrain_counter += j

				n += j

			# retrain on observed (x, a, r) triplets.
			nn.train(observed_triplets)
			# change n_epochs and next_retrain value depending on measured non-stationarity :)
			next_retrain = min(next_retrain * K, num_epochs)

		t1 = time.time()
		exec_time[i] = t1-t0
		if verbose:
			print("Algorithm: Variable Deep Thompson Sampling. Run " + str(i+1) + "/" + str(NRuns) + " completed. Final Regret: " + str(maxG[NSteps-1]-G[i, NSteps-1]) + " Average Best-Action %: " + str(sum(BA[i, :])/NSteps) + ". Execution time: " + str(exec_time[i]) + " sec.")

	# Compute average results
	for i in range(NSteps):
		avg_regret[i] = maxG[i] - sum(G[:, i])/NRuns
		avg_BA[i] = sum(BA[:, i])/NRuns

	return avg_regret, avg_BA, sum(exec_time)/NRuns