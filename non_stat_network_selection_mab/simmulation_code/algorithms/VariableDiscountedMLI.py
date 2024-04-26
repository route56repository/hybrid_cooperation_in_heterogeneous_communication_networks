import numpy as np
import time

# MLI algorithm with Variable Discounted variation.
def VariableDiscountedMLI(scenario, maxG):
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

        rewards = np.zeros(m)

        # initial interval between measures values (from MLI)
        interval = d2*nu

        # initial discount factor value
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
        
        # Initial measures
        for k in range(m):
            for j in range(d1):
                for step in range(nm):
                    nk[k] = nk[k]*discount_factor[k]
                    nk[k] += 1
                    w = scenario.generate_reward(k, n)
                    rewards[k] = rewards[k]*discount_factor[k]
                    rewards[k] += w
                    mu[k] += rewards[k]/nk[k]

        n += m*nm*d1
        next_measure = interval+n

        while n < NSteps:
            # measure
            if n >= next_measure:
                k = np.argmin(nk)
                j = 0
                while (j<nm) and ((j+n)<NSteps):
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
                        accumulated_m0[k] += 1
                        accumulated_m1[k] = 0
                        if accumulated_m0[k] == m0 and discount_factor[k] != discount_max:
                            discount_factor[k] = round(discount_factor[k]+discount_step, 2)
                            accumulated_m0[k] = 0

                    previous_sign[k] = sign

                    for arm in range(m):
                        nk[arm] = nk[arm]*discount_factor[arm]
                        rewards[arm] = rewards[arm]*discount_factor[arm]

                    nk[k] += 1
                    rewards[k] += w

                    j += 1

                for arm in range(m):
                    mu[arm] = rewards[arm]/(max(nk[arm], 0.01))

                G[i, n:(n+j)] = G[i, n-1]
                interval = min(np.ceil(interval + np.log(interval)), 400)

                Nk = Nk*discount_factor
                Nk[k] += 1

                n += j
                next_measure = n+interval

                N = sum(Nk)

            # use
            else:
                k = np.argmax(mu)
                j = 0
                g = 0
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
                        accumulated_m0[k] += 1
                        accumulated_m1[k] = 0
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
                    mu[arm] = rewards[arm]/(max(nk[arm], 0.01))

                if k == scenario.get_best_arm(n):
                    BA[i, n:(n+j)] = 1

                G[i, n:(n+j)] = g + G[i, n-1]
                n += j

                Nk = Nk*discount_factor
                Nk[k] += 1
                
                N = sum(Nk)
            
        t1 = time.time()
        exec_time[i] = t1-t0

    # average the results
    for i in range(NSteps):
        avg_regret[i] = maxG[i] - sum(G[:, i])/NRuns
        avg_BA[i] = sum(BA[:, i])/NRuns
    return avg_regret, avg_BA, sum(exec_time)/NRuns