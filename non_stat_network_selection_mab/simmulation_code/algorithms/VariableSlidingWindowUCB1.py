import numpy as np
import time

# UCB1 algorithm with Variable Sliding Window variation
def VariableSlidingWindowUCB1(scenario, maxG):
	# Get necessary parameters from the environment and simmulation
	NSteps = scenario.NSteps
	NRuns = scenario.NRuns
	m = scenario.m
	nu = scenario.nu

	# Variable to store results
	avg_regret = np.zeros(NSteps)
	G = np.zeros((NRuns, NSteps))

	exec_time = np.zeros(NRuns)

	BA = np.zeros((NRuns, NSteps))
	avg_BA = np.zeros(NSteps)


	for i in range(NRuns):
		mu = np.zeros(m)

		n = 0 # initial step
		nk = np.zeros(m) # num of steps arm k was selected
		N = 0 # initial num of choices done
		Nk = np.zeros(m) # num of times arm k has been chosen

		selected_arms = np.zeros(NSteps)
		rewards = np.zeros(NSteps)
		rewards_window = np.zeros(NSteps)

		best_action = 0

		I = np.zeros(m)

		# Sliding Window values
		window_size = 5000
		window_step = 500
		window_min = 500
		window_max = 20000
		# Parameters to count consecutive sign values
		m0 = 5
		m1 = 10
		accumulated_m0 = 0
		accumulated_m1 = 0
		previous_sign = "-"

		t0 = time.time()

		while n < NSteps:
			I = mu + np.sqrt((2*np.log(max(N, 1)))/(Nk+0.01))
			k = int(np.argmax(I))

			j = 0
			g = 0
			while (j<nu) and ((j+n)<NSteps):
				w = scenario.generate_reward(k, n)

				sign = "+"
				if w < mu[k]:
					sign = "-"

				# Modify sliding window value
				if previous_sign == sign:
					accumulated_m1 += 1
					accumulated_m0 = 0

					if accumulated_m1 == m1 and window_size != window_min:
						window_size -= window_step

						for step in range(window_step):
							if (n+j) > (window_size + step):
								nk[int(selected_arms[n+j-window_size-step])] -= 1
								rewards_window[int(selected_arms[n+j-window_size-step])] -= rewards[n+j-window_size-step]

						for step in range(int(window_step/nu)):
							if (n+j) > (window_size + step*nu):
								Nk[int(selected_arms[n+j-window_size-step*nu])] -= 1
						accumulated_m1 = 0
				else:
					accumulated_m0 += 1
					accumulated_m1 = 0

					if accumulated_m0 == m0 and window_size != window_max:
						window_size += window_step

						for step in range(window_step):
							if (n+j) > (window_size - step):
								nk[int(selected_arms[n+j-window_size+step])] += 1
								rewards_window[int(selected_arms[n+j-window_size+step])] += rewards[n+j-window_size+step]

						for step in range(int(window_step/nu)):
							if (n+j) > (window_size - step*nu):
								Nk[int(selected_arms[n+j-window_size+step*nu])] += 1
						accumulated_m0 = 0

				previous_sign = sign

				selected_arms[n+j] = k
				nk[k] += 1
				if (n+j) >= window_size:
					nk[int(selected_arms[n+j-window_size])] -= 1
				
				rewards[n+j] = w
				rewards_window[k] += w

				if (n+j) >= window_size:
					rewards_window[int(selected_arms[n+j-window_size])] -= rewards[n+j-window_size]

				g += w
				j += 1

			for arm in range(m):
				mu[arm] = rewards_window[arm]/max(nk[arm], 0.01)

			if k == scenario.get_best_arm(n):
				BA[i, n:(n+j)] = 1

			G[i, n:(n+j)] = g + G[i, n-1]

			n += j

			Nk[k] += 1
			if n > window_size:
				Nk[int(selected_arms[n+j-window_size])] -= 1

			N = sum(Nk)

			for arm in range(m):
				if Nk[arm]<0:
					Nk[np.argmax(Nk)] += Nk[arm]
					Nk[arm] = 0

		t1 = time.time()
		exec_time[i] = t1-t0

	# average the results
	for i in range(NSteps):
		avg_regret[i] = maxG[i] - sum(G[:, i])/NRuns
		avg_BA[i] = sum(BA[:, i])/NRuns

	return avg_regret, avg_BA, sum(exec_time)/NRuns