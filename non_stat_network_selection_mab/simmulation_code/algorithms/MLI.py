import numpy as np
import time

# MLI algorithm
def MLI(scenario, maxG):
    # Get necessary parameters from the environment and simmulation
    NSteps = scenario.NSteps
    NRuns = scenario.NRuns
    m = scenario.m
    nu = scenario.nu
    nm = scenario.nm

    # MLI Constants
    d1 = 5
    d2 = 5

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
        N = 0 # initial num of choices done
        Nk = np.zeros(m) # num of times arm k has been chosen

        interval = d2*nu # interval between measures

        t0 = time.time()
        
        # Initial measures
        for k in range(m):
            for j in range(d1):
                for step in range(nm):
                    nk[k] += 1
                    w = scenario.generate_reward(k, n)
                    mu[k] += 1/nk[k]*(w-mu[k])

        n += m*nm*d1
        next_measure = interval+n

        while n < NSteps:
            # measure during nm steps
            if n >= next_measure:
                k = np.argmin(nk)
                j = 0
                while (j<nm) and ((j+n)<NSteps):
                    w = scenario.generate_reward(k, n)
                    nk[k] += 1
                    mu[k] += 1/nk[k]*(w-mu[k])
                    j += 1

                G[i, n:(n+j)] = G[i, n-1]
                interval = np.ceil(interval + np.log(interval))
                Nk[k] += 1
                n += j
                next_measure = n+interval
                N += 1

            # use during nu steps
            else:
                k = np.argmax(mu)
                j = 0
                g = 0
                while (j<nu) and ((j+n)<NSteps):
                    w = scenario.generate_reward(k, n)
                    nk[k] += 1
                    mu[k] += 1/(nk[k])*(w-mu[k])
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

    # compute average rewards
    for i in range(NSteps):
        avg_regret[i] = maxG[i] - sum(G[:, i])/NRuns
        avg_BA[i] = sum(BA[:, i])/NRuns
    return avg_regret, avg_BA, sum(exec_time)/NRuns