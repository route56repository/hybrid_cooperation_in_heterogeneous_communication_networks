import numpy as np
import time

# Gradient Bandit algorithm
def GradientBandit(scenario, maxG):
    # Get necessary parameters from the environment and simmulation
    NSteps = scenario.NSteps
    NRuns = scenario.NRuns
    m = scenario.m
    nu = scenario.nu

    # Gradient Bandit constants
    alpha = 0.1

    # Parameters to store obtained rewards
    avg_regret = np.zeros(NSteps)
    G = np.zeros((NRuns, NSteps))

    exec_time = np.zeros(NRuns)

    BA = np.zeros((NRuns, NSteps))
    avg_BA = np.zeros(NSteps)


    for i in range(NRuns):
        
        avg_reward = 0
        n = 0

        probabilities = np.zeros(m)
        H = np.zeros(m)

        t0 = time.time()
        
        while n < NSteps:
            # compute probability of selecting an arm given H
            probabilities = softmax(H, m)
            
            k = select_arm(probabilities, m)
            
            j = 0
            g = 0
            # always uses during nu steps
            while (j<nu) and ((j+n)<NSteps):
                w = scenario.generate_reward(k, n)

                avg_reward += 1/max((n+j), 1)*(w-avg_reward)

                # update the value of H
                H[k] = H[k] + alpha*(w-avg_reward)*(1-probabilities[k])
                for arm in range(m):
                    if arm != k:
                        H[arm] = H[arm] - alpha*(w-avg_reward)*probabilities[arm]

                g += w
                j += 1

            if k == scenario.get_best_arm(n):
                BA[i, n:(n+j)] = 1

            G[i, n:(n+j)] = g + G[i, n-1]

            n += j 
            
        t1 = time.time()
        exec_time[i] = t1-t0

    # compute average rewards
    for i in range(NSteps):
        avg_regret[i] = maxG[i] - sum(G[:, i])/NRuns
        avg_BA[i] = sum(BA[:, i])/NRuns
    return avg_regret, avg_BA, sum(exec_time)/NRuns

def select_arm(probabilities, m):
    prob = np.random.random()
    accum_prob = 0
    for k in range(m):
        if (accum_prob <= prob) and ((accum_prob + probabilities[k]) > prob):
            return k
        accum_prob += probabilities[k]
    return k

def softmax(H, m):
    probabilities = np.zeros(m)
    denominator = 0
    for k in range(m):
        denominator += np.exp(H[k]-np.max(H))
    for k in range(m):
        probabilities[k] = np.exp(H[k]-np.max(H))/denominator
    return probabilities