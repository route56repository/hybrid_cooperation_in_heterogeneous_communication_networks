import numpy as np
import time

def SlidingWindowUCB1(scenario, maxG, verbose=True):
    # Get necessary parameters from the environment and simmulation
	NSteps = scenario.NSteps
	NRuns = scenario.NRuns
	m = scenario.m
	nu = scenario.nu

	# Parameters to store obtained results
	avg_regret = np.zeros(NSteps)
	G = np.zeros((NRuns, NSteps))

	exec_time = np.zeros(NRuns)

	BA = np.zeros((NRuns, NSteps))
	avg_BA = np.zeros(NSteps)

	# Parameters for Sliding Window UCB1
	window_size = 7000

	for i in range(NRuns):
        # Initialize values
		mu = np.zeros(m)

		n = 0
		nk = np.zeros(m)
		N = 0
		Nk = np.zeros(m)

		selected_arms = np.zeros(NSteps)
		rewards = np.zeros(NSteps)
		rewards_window = np.zeros(NSteps)

		best_action = 0

		I = np.zeros(m)
		t0 = time.time()

        # Start main loop
		while n < NSteps:
            # Select arm according to mean estimate and uncertainty itnerval
			I = mu + np.sqrt((2*np.log(max(N, 1)))/(Nk+0.01))
			k = int(np.argmax(I))

            # Get rewards for selected arm
			j = 0
			g = 0
			while (j<nu) and ((j+n)<NSteps):
				w = scenario.generate_reward(k, n+j)

				selected_arms[n+j] = k
				nk[k] += 1
                # Move window...
				if (n+j) >= window_size:
					nk[int(selected_arms[n+j-window_size])] -= 1

				rewards[n+j] = w
				rewards_window[k] += w

                # Move window...
				if (n+j) >= window_size:
					rewards_window[int(selected_arms[n+j-window_size])] -= rewards[n+j-window_size]

				g += w
				j += 1

            # Re-compute mean estimate
			for arm in range(m):
				mu[arm] = rewards_window[arm]/max(nk[arm], 0.01)

			if k == scenario.get_best_arm(n):
				BA[i, n:(n+j)] = 1

			G[i, n:(n+j)] = g + G[i, n-1]

			n += j

			Nk[k] += 1

            # Move window...
			if n > window_size:
				Nk[int(selected_arms[n+j-window_size])] -= 1

			N = sum(Nk)

		t1 = time.time()
		exec_time[i] = t1-t0

		if verbose:
			print("Algorithm: Sliding Window UCB1. Run " + str(i+1) + "/" + str(NRuns) + " completed. Final Regret: " + str(maxG[NSteps-1]-G[i, NSteps-1]) + " Average Best-Action %: " + str(sum(BA[i, :])/NSteps) + ". Execution time: " + str(exec_time[i]) + " sec.")
    # Compute average rewards
	for i in range(NSteps):
		avg_regret[i] = maxG[i] - sum(G[:, i])/NRuns
		avg_BA[i] = sum(BA[:, i])/NRuns

	return avg_regret, avg_BA, sum(exec_time)/NRuns