import numpy as np
import time

# UCB1 algorithm with Variable Discounted variation
def VariableDiscountedUCB1(scenario, maxG):
	# Get necessary parameters from the environment and simmulation
	NSteps = scenario.NSteps
	NRuns = scenario.NRuns
	m = scenario.m
	nu = scenario.nu

	# Variables to store results
	avg_regret = np.zeros(NSteps)
	G = np.zeros((NRuns, NSteps))

	exec_time = np.zeros(NRuns)

	BA = np.zeros((NRuns, NSteps))
	avg_BA = np.zeros(NSteps)

	i = 0

	for i in range(NRuns):
		mu = np.zeros(m)

		n = 0 # initial step
		nk = np.zeros(m) # num of steps arm k was selected
		N = 0 # initial num of choices done
		Nk = np.zeros(m) # num of times arm k has been chosen

		rewards = np.zeros(m)

		I = np.zeros(m)

		# discount factor values
		discount_factor = np.zeros(m) + 0.7
		discount_step = 0.05
		discount_max = 1
		discount_min = 0
		# parameters to count consecutive sign values
		m0 = 5
		m1 = 10
		accumulated_m0 = np.zeros(m)
		accumulated_m1 = np.zeros(m)
		previous_sign = []
		for arm in range(m):
			previous_sign.append("-")

		t0 = time.time()

		while n < NSteps:
			I = mu + np.sqrt((2*np.log(max(N, 1)))/(Nk+0.01))
			k = int(np.argmax(I))

			j = 0
			g = 0

			# UCB1 algorithm only uses
			while (j<nu) and ((j+n)<NSteps):
				w = scenario.generate_reward(k, n)

				sign = "+"
				if w < mu[k]:
					sign = "-"

				# change discount factor value
				if previous_sign[k] == sign:
					accumulated_m1[k] += 1
					accumulated_m0[k] = 0

					if accumulated_m1[k] == m1 and discount_factor[k] != discount_min:
						discount_factor[k] = round(discount_factor[k]-discount_step, 2)
						accumulated_m1[k] = 0

				else:
					accumulated_m1[k] = 0
					accumulated_m0[k] += 1

					if accumulated_m0[k] == m0 and discount_factor[k] != discount_max:
						discount_factor[k] = round(discount_factor[k]+discount_step, 2)
						accumulated_m0[k] = 0

				previous_sign[k] = sign

				for arm in range(m):
					nk[arm] = nk[arm]*discount_factor[arm]
					rewards[arm] = rewards[arm]*discount_factor[arm]

				nk[k] += 1
				rewards[k] += w

				g += w
				j += 1

			for arm in range(m):
				mu[arm] = rewards[arm]/max(nk[arm], 0.01)

			if k == scenario.get_best_arm(n):
				BA[i, n:(n+j)] = 1

			G[i, n:(n+j)] = g + G[i, n-1]

			n += j

			Nk = Nk*discount_factor
			Nk[k] += 1

			N = sum(Nk)

		t1 = time.time()
		exec_time[i] = t1-t0

	for i in range(NSteps):
		avg_regret[i] = maxG[i] - sum(G[:, i])/NRuns
		avg_BA[i] = sum(BA[:, i])/NRuns

	return avg_regret, avg_BA, sum(exec_time)/NRuns