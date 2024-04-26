__author__ = 'lekesen'

# Import libraries & classes
from algorithms.auxiliar.ContextualNN_4 import ContextualNN4
import numpy as np
from scipy.stats import invgauss
from scipy.stats import beta as beta_distr
import time

'''
Deep Contextual MAB says...
In the below experiments for the contextual models (including the epsilon-greedy and fixed dropout rate bandits),
we use a NN with 2 hidden layers with 256 units each and ReLU activation function. All networks are trained using
the Adam optimizer with initial learning rate = 0.001. We retrain the contextual model on an exponential scale:
K = 2 and N = 128
'''

def DeepThompsonSampling3(scenario, maxG, verbose=True):
	# Get necessary parameters from the environment and simmulation
	NSteps = scenario.NSteps
	NRuns = scenario.NRuns
	m = scenario.m
	nu = scenario.nu
	nm = scenario.nm # check if we could use them

	#history = open("history.txt", "w")

	# Parameters to save obtained results (regret, execution time & BA)
	avg_regret = np.zeros(NSteps)
	G = np.zeros((NRuns, NSteps))

	exec_time = np.zeros(NRuns)

	BA = np.zeros((NRuns, NSteps))
	avg_BA = np.zeros(NSteps)

	# Parameters for Deep Thompson Sampling
	K = 1.05
	N = 15

	# Loop for each run
	for i in range(NRuns):
		# Initialize values
		n = 0 # initial step

		next_retrain = N

		# Initialize neural network
		nn = []
		for arm in range(m):
			nn_th = ContextualNN4()
			nn_dl = ContextualNN4()
			nn_lr = ContextualNN4()

			nn_th.create_model(m)
			nn_dl.create_model(m)
			nn_lr.create_model(m)

			tmp_nn = []
			tmp_nn.append(nn_th)
			tmp_nn.append(nn_dl)
			tmp_nn.append(nn_lr)

			nn.append(tmp_nn)

		t0 = time.time()

		# Get initial measures to start training NN...
		initial_rewards = []
		for arm in range(m):
			th = []
			dl = []
			lr = []
			tmp = []
			tmp.append(th)
			tmp.append(dl)
			tmp.append(lr)
			initial_rewards.append(tmp)


		for repetition in range(1):
			for arm in range(m):
				g = 0
				step = 0
				while step < nm:
					th = scenario.getTH(arm, n)
					dl = scenario.getDL(arm, n)
					lr = scenario.getLR(arm, n)

					r = scenario.getQoE(th, lr, dl, n+step)

					if arm == scenario.get_best_arm(n+step):
						BA[i, n:(n+step)] = 1

					(initial_rewards[arm])[0].append([1, th])
					(initial_rewards[arm])[1].append([1, dl*100])
					(initial_rewards[arm])[2].append([1, lr*100])
					step += 1

				G[i, n:(n+step)] = g + G[i, n-1]
				n += step

		for arm in range(m):
			for nw_param in range(3):
				train_nn = (nn[arm])[nw_param]
				train_nn.train((initial_rewards[arm])[nw_param], num_epochs=15)

		# Start main loop
		while n<NSteps:
			retrain_counter = 0
			observed_triplets = []

			for arm in range(m):
				th = []
				dl = []
				lr = []
				tmp = []
				tmp.append(th)
				tmp.append(dl)
				tmp.append(lr)
				observed_triplets.append(tmp)

			while retrain_counter<next_retrain and n<NSteps:
				x = scenario.getContext(n)

				# Compute predicted reward for each arm
				mean_th = np.zeros(m)
				mean_dl = np.zeros(m)
				mean_lr = np.zeros(m)

				var_th = np.zeros(m)
				var_dl = np.zeros(m)
				var_lr = np.zeros(m)

				pred_th = np.zeros(m)
				pred_dl = np.zeros(m)
				pred_lr = np.zeros(m)
				r = np.zeros(m)
				for arm in range(m):
					mean_th[arm], var_th[arm] = (nn[arm])[0].predict(arm)
					pred_th[arm] = np.random.normal(mean_th[arm], var_th[arm]) #invgauss.rvs((mean_th[arm])/gamma_th, scale = gamma_th)*10
					while pred_th[arm] < 0:
						pred_th[arm] = np.random.normal(mean_th[arm], var_th[arm])

					mean_dl[arm], var_dl[arm] = (nn[arm])[1].predict(arm)
					pred_dl[arm] = np.random.normal(mean_dl[arm], var_dl[arm])/100 # max(np.random.normal(mean_dl[arm], var_dl[arm]), 0)

					while pred_dl[arm] < 0:
						pred_dl[arm] = np.random.normal(mean_dl[arm], var_dl[arm])/100

					mean_lr[arm], var_lr[arm] = (nn[arm])[2].predict(arm)
					pred_lr[arm] = np.random.normal(mean_lr[arm], var_lr[arm])/100 # pred_lr[arm] = mean_lr[arm]*beta_distr.rvs(self.alphaLR, self.betaLR)
					while pred_lr[arm] < 0 or pred_lr[arm] > 1:
						pred_lr[arm] = np.random.normal(mean_lr[arm], var_lr[arm])/100
					
					r[arm] = scenario.getQoE(pred_th[arm], pred_dl[arm], pred_lr[arm], n)

				# Select arm with best predicted reward
				k = np.argmax(r)
				#k = scenario.get_best_arm(n)
				# Get rewards for selected arm (nu at the moment)
				j = 0
				g = 0
				while (j<nu) and ((j+n)<NSteps):
					th = scenario.getTH(k, n+j)
					dl = scenario.getDL(k, n+j)
					lr = scenario.getLR(k, n+j)

					w = scenario.getQoE(th, lr, dl, n+j)
					g += w
					j += 1

					(observed_triplets[k])[0].append([1, th])
					(observed_triplets[k])[1].append([1, dl*100])
					(observed_triplets[k])[2].append([1, lr*100])

				# Save observed rewards & info
				if k == scenario.get_best_arm(n):
					BA[i, n:(n+j)] = 1

				G[i, n:(n+j)] = g + G[i, n-1]

				retrain_counter += j

				n += j

			for repetition in range(1):
				for arm in range(m):
					g = 0
					step = 0
					while step < nm and (n+step) != NSteps:
						th = scenario.getTH(arm, n)
						dl = scenario.getDL(arm, n)
						lr = scenario.getLR(arm, n)

						r = scenario.getQoE(th, lr, dl, n+step)

						if arm == scenario.get_best_arm(n+step):
							BA[i, n:(n+step)] = 1

						(observed_triplets[arm])[0].append([1, th])
						(observed_triplets[arm])[1].append([1, dl*100])
						(observed_triplets[arm])[2].append([1, lr*100])

						step += 1

					G[i, n:(n+step)] = g + G[i, n-1]
					n += step
			# retrain on observed (x, a, r) triplets.
			for arm in range(m):
				for nw_param in range(3):
					train_nn = (nn[arm])[nw_param]
					train_nn.train((initial_rewards[arm])[nw_param], num_epochs=10)
			# change n_epochs and next_retrain value depending on measured non-stationarity :)
			next_retrain = min(next_retrain * K, 5000)

		t1 = time.time()
		exec_time[i] = t1-t0
		if verbose:
			print("Algorithm: Deep Thompson Sampling 3. Run " + str(i+1) + "/" + str(NRuns) + " completed. Final Regret: " + str(maxG[NSteps-1]-G[i, NSteps-1]) + " Average Best-Action %: " + str(sum(BA[i, :])/NSteps) + ". Execution time: " + str(exec_time[i]) + " sec.")
			#history.write("Algorithm: Deep Thompson Sampling. Run " + str(i+1) + "/" + str(NRuns) + " completed. Final Regret: " + str(maxG[NSteps-1]-G[i, NSteps-1]) + " Average Best-Action %: " + str(sum(BA[i, :])/NSteps) + ". Execution time: " + str(exec_time[i]) + " sec.")
	# Compute average results
	for i in range(NSteps):
		avg_regret[i] = maxG[i] - sum(G[:, i])/NRuns
		avg_BA[i] = sum(BA[:, i])/NRuns

	#history.close()
	return avg_regret, avg_BA, sum(exec_time)/NRuns