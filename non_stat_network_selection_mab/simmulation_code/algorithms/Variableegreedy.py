import numpy as np
import time

# e-greedy with an adaptive variation
def Variableegreedy(scenario, maxG):
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

    for i in range(NRuns):
        
        mu = np.zeros(m)  # maybe we need to change initial values

        n = 0 # initial step
        nk = np.zeros(m) # num of steps arm k was selected

        # epsilon values
        epsilon = 0.1
        epsilon_step = 0.05
        epsilon_max = 0.5
        epsilon_min = 0

        # discount factor values
        discount_factor = 0.7
        discount_step = 0.05
        discount_max = 1
        discount_min = 0.05

        # parameters to count consecutive sign values
        m0 = 5
        m1 = 10
        accumulated_m0 = 0
        accumulated_m1 = 0

        previous_sign = "-"

        rewards = np.zeros(m)

        t0 = time.time()
        
        while n < NSteps:
            I = np.argmax(mu)
            if np.random.rand() > epsilon:
                k = I
            else:
                randIndex = np.random.randint(m-1)
                k = randIndex + (randIndex >= I)
            
            j = 0
            g = 0
            # only uses
            while (j<nu) and ((j+n)<NSteps):
                w = scenario.generate_reward(k, n)

                sign = "+"
                if w < mu[k]:
                    sign = "-"

                # change discount_factor and epsilon values
                if previous_sign == sign:
                    accumulated_m1 += 1
                    accumulated_m0 = 0
                    if accumulated_m1 == m1:
                        if discount_factor != discount_min:
                            discount_factor = round(discount_factor-discount_step, 2)
                        if epsilon != epsilon_max:
                            epsilon = round(epsilon+epsilon_step, 2)
                        accumulated_m1 = 0
                else:
                    accumulated_m0 += 1
                    accumulated_m1 = 0
                    if accumulated_m0 == m0:
                        if discount_factor != discount_max:
                            discount_factor = round(discount_factor+discount_step, 2)
                        if epsilon != epsilon_min:
                            epsilon = round(epsilon-epsilon_step, 2)
                        accumulated_m0 = 0

                previous_sign = sign

                nk = nk*discount_factor
                nk[k] += 1

                rewards = rewards*discount_factor
                rewards[k] += w

                g += w
                j += 1

            for arm in range(m):
                mu[arm] = rewards[arm]/max(nk[arm], 0.01)

            if k == scenario.get_best_arm(n):
                BA[i, n:(n+j)] = 1

            G[i, n:(n+j)] = g + G[i, n-1]

            n += j       
            
        t1 = time.time()
        exec_time[i] = t1-t0

    # average the results
    for i in range(NSteps):
        avg_regret[i] = maxG[i] - sum(G[:, i])/NRuns
        avg_BA[i] = sum(BA[:, i])/NRuns
    return avg_regret, avg_BA, sum(exec_time)/NRuns