import numpy as np
import time


def egreedy(scenario, maxG, verbose=True):
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

    # Parameters for e-greedy.
    epsilon = 0.05

    # Loop for each run
    for i in range(NRuns):
        mu = np.zeros(m)  # maybe we need to change initial values

        n = 0 # initial step
        nk = np.zeros(m) # num of steps arm k was selected

        t0 = time.time()

        # Main loop
        while n < NSteps:
            # Exploit with prob (1-e) and explore with prob (e)
            I = np.argmax(mu)
            if np.random.rand() > epsilon:
                k = I
            else:
                randIndex = np.random.randint(m-1)
                k = randIndex + (randIndex >= I)

            # Get rewards for chosen arm
            j = 0
            g = 0
            while (j<nu) and ((j+n)<NSteps):
                w = scenario.generate_reward(k, n+j)
                nk[k] += 1
                mu[k] += 1/(nk[k])*(w - mu[k])

                g += w
                j += 1

            if k == scenario.get_best_arm(n):
                BA[i, n:(n+j)] = 1

            G[i, n:(n+j)] = g + G[i, n-1]

            n += j


        t1 = time.time()
        exec_time[i] = t1-t0
        if verbose:
            print("Algorithm: e-greedy. Run " + str(i+1) + "/" + str(NRuns) + " completed. Final Regret: " + str(maxG[NSteps-1]-G[i, NSteps-1]) + " Average Best-Action %: " + str(sum(BA[i, :])/NSteps) + ". Execution time: " + str(exec_time[i]) + " sec.")
    # Compute average results
    for i in range(NSteps):
        avg_regret[i] = maxG[i] - sum(G[:, i])/NRuns
        avg_BA[i] = sum(BA[:, i])/NRuns
    return avg_regret, avg_BA, sum(exec_time)/NRuns