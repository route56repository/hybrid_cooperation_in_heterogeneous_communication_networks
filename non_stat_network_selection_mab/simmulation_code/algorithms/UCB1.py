import numpy as np
import time

# UCB1 algorithm
def UCB1(scenario, maxG):
    # Get necessary parameters from the environment and simmulation
    NSteps = scenario.NSteps
    NRuns = scenario.NRuns
    m = scenario.m
    nu = scenario.nu

    # Parameters to store results
    avg_regret = np.zeros(NSteps)
    G = np.zeros((NRuns, NSteps))

    exec_time = np.zeros(NRuns)

    BA = np.zeros((NRuns, NSteps))
    avg_BA = np.zeros(NSteps)
    

    for i in range(NRuns):
        
        mu = np.zeros(m)  # maybe we need to change initial values

        n = 0 # initial step
        nk = np.zeros(m) # num of steps arm k was selected
        N = 0 # initial num of choices done
        Nk = np.zeros(m) # num of times arm k has been chosen

        I = np.zeros(m)

        t0 = time.time()
        
        # Loop for each run
        while n < NSteps:
            # Select arm according to mean estimate & uncertainty interval
            I = mu + np.sqrt((2*np.log(max(N, 1)))/(Nk+0.01))
            k = np.argmax(I) 
            
            # Get rewards for selected arm
            j = 0
            g = 0
            # UCB1 algorithm only uses
            while (j<nu) and ((j+n)<NSteps):
                w = scenario.generate_reward(k, n)
                nk[k] += 1
                mu[k] += 1/(nk[k])*(w - mu[k])

                g += w
                j += 1

            if k == scenario.get_best_arm(n):
                BA[i, n:(n+j)] = 1

            G[i, n:(n+j)] = g + G[i, n-1]

            n += j
            Nk[k] += 1
            N += 1
            
        t1 = time.time()
        exec_time[i] = t1-t0

    # Compute average rewards
    for i in range(NSteps):
        avg_regret[i] = maxG[i] - sum(G[:, i])/NRuns
        avg_BA[i] = sum(BA[:, i])/NRuns
    return avg_regret, avg_BA, sum(exec_time)/NRuns