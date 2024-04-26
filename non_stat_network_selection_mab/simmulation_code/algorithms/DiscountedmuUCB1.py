import numpy as np
import time

# muUCB1 algorithm with the 'Discounted' variation
def DiscountedmuUCB1(scenario, maxG):
    # Get necessary parameters from the environment and simmulation
    NSteps = scenario.NSteps
    NRuns = scenario.NRuns
    m = scenario.m
    nu = scenario.nu
    nm = scenario.nm

    # Discounted Constants
    discount_factor = 0.9

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

        rewards = np.zeros(m)

        I = np.zeros(m)

        t0 = time.time()
        

        while n < NSteps:
            # Use estimate and uncertainty interval
            I = mu + np.sqrt((2*np.log(max(N, 1)))/(Nk+0.01))
            k = np.argmax(I) 
            
            # use during nu steps
            if k == np.argmax(mu):
                j = 0
                g = 0
                while (j<nu) and ((j+n)<NSteps):
                    w = scenario.generate_reward(k, n)
                    
                    nk = nk*discount_factor
                    nk[k] += 1

                    rewards = rewards*discount_factor
                    rewards[k] += w

                    g += w
                    j += 1

                # update mean estimate
                for arm in range(m):
                    mu[arm] = rewards[arm]/max(nk[arm], 0.01)

                if k == scenario.get_best_arm(n):
                    BA[i, n:(n+j)] = 1

                G[i, n:(n+j)] = g + G[i, n-1]

                n += j
                
                Nk = Nk*discount_factor
                Nk[k] += 1

                N = sum(Nk)

            # measure during nm steps
            else:
                j = 0
                while (j<nm) and ((j+n)<NSteps):
                    w = scenario.generate_reward(k, n)

                    nk = nk*discount_factor
                    nk[k] += 1

                    rewards = rewards*discount_factor
                    rewards[k] += w

                    j += 1

                # update mean estimate
                for arm in range(m):
                    mu[arm] = rewards[arm]/max(nk[arm], 0.01)

                G[i, n:(n+j)] = G[i, n-1]

                n += j

                Nk = Nk*discount_factor
                Nk[k] += 1

                N = sum(Nk)
            
        t1 = time.time()
        exec_time[i] = t1-t0

    # Compute average rewards
    for i in range(NSteps):
        avg_regret[i] = maxG[i] - sum(G[:, i])/NRuns
        avg_BA[i] = sum(BA[:, i])/NRuns
    return avg_regret, avg_BA, sum(exec_time)/NRuns