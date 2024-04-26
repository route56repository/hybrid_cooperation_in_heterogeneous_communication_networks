__author__ = 'lekesen'

# Import libraries & classes
from algorithms.auxiliar.ContextualNN_2 import ContextualNN2
import numpy as np
import time


def DeepThompsonSampling(scenario, maxG, verbose=True):
	# Get necessary parameters from the environment and simmulation
	NSteps = scenario.NSteps
	NRuns = scenario.NRuns
	m = scenario.m
	nu = scenario.nu
	nm = scenario.nm # check if we could use them

	history = open("history.txt", "w")

	# Parameters to save obtained results (regret, execution time & BA)
	avg_regret = np.zeros(NSteps)
	G = np.zeros((NRuns, NSteps))

	#reward = np.zeros((NRuns, NSteps))
	#avg_reward = np.zeros(NSteps)

	exec_time = np.zeros(NRuns)

	BA = np.zeros((NRuns, NSteps))
	avg_BA = np.zeros(NSteps)

	# Parameters for Deep Thompson Sampling
	K = 1.05
	N = 15

	num_repetitions = 5

	# Loop for each run
	for i in range(NRuns):
		# Initialize values
		n = 0 # initial step

		next_retrain = N

		# Initialize neural network
		nn = ContextualNN2()
		nn.create_model(m)

		t0 = time.time()

		# Get initial measures to start training NN...
		initial_rewards = []
		for repetition in range(1):
			for arm in range(m):
				g = 0
				step = 0
				while step < nm:
					x = scenario.getContext(n)
					r = scenario.generate_reward(arm, n+step)

					if arm == scenario.get_best_arm(n+step):
						BA[i, n:(n+step)] = 1

					initial_rewards.append([x, arm, r])
					
					#reward[i, n+step] = 0

					step += 1

				G[i, n:(n+step)] = g + G[i, n-1]
				n += step

		nn.train(initial_rewards, num_epochs=50) # try to change epochs ?
		# Start main loop
		while n<NSteps:
			retrain_counter = 0
			observed_triplets = []
			while retrain_counter<next_retrain and n<NSteps:
				x = scenario.getContext(n)

				# Compute predicted reward for each arm
				r = np.zeros(m)
				mean = np.zeros(m)
				var = np.zeros(m)
				
				for arm in range(m):
					for repe in range(num_repetitions):
						mean[arm], var[arm] = nn.predict(x, arm)
						r[arm] += np.random.normal(mean[arm], np.sqrt(var[arm])/20) # remove np.sqrt
					r[arm] = r[arm]/num_repetitions

				
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
						history.write('Step ' + str(n+j) + '\t Selected arm: ' + str(k) + '\t Predicted reward: ' + str(r).replace('\n', ' ') + '\t Predicted mean: ' + str(mean).replace('\n', ' ') + '\t Predicted var: ' + str(var).replace('\n', ' ') + '\t Best arm: ' + str(scenario.get_best_arm(n)) + '\n')
				# Save observed rewards & info
				if k == scenario.get_best_arm(n):
					BA[i, n:(n+j)] = 1

				G[i, n:(n+j)] = g + G[i, n-1]

				retrain_counter += j

				n += j

			# retrain on observed (x, a, r) triplets.
			nn.train(observed_triplets)
			# change n_epochs and next_retrain value depending on measured non-stationarity
			next_retrain = min(next_retrain * K, 3000)

		t1 = time.time()
		exec_time[i] = t1-t0
		if verbose:
			print("Algorithm: Deep Thompson Sampling. Run " + str(i+1) + "/" + str(NRuns) + " completed. Final Regret: " + str(maxG[NSteps-1]-G[i, NSteps-1]) + " Average Best-Action %: " + str(sum(BA[i, :])/NSteps) + ". Execution time: " + str(exec_time[i]) + " sec.")
			history.write("Algorithm: Deep Thompson Sampling. Run " + str(i+1) + "/" + str(NRuns) + " completed. Final Regret: " + str(maxG[NSteps-1]-G[i, NSteps-1]) + " Average Best-Action %: " + str(sum(BA[i, :])/NSteps) + ". Execution time: " + str(exec_time[i]) + " sec.")
	# Compute average results
	for i in range(NSteps):
		avg_regret[i] = maxG[i] - sum(G[:, i])/NRuns
		avg_BA[i] = sum(BA[:, i])/NRuns

	history.close()
	return avg_regret, avg_BA, sum(exec_time)/NRuns