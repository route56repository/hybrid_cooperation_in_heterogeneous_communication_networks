import numpy as np
import time

# Gradient Bandit algorithm programmed to be adaptive
def VariableGradientBandit(scenario, maxG):
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
        
        avg_reward = 0
        
        n = 0
        nk = np.zeros(m)
        probabilities = np.zeros(m)
        H = np.zeros(m)

        rewards = 0
        rewards_arm = np.zeros(m)

        # alpha (from Gradient Bandit) values
        alpha = 0.2
        alpha_step = 0.05
        alpha_max = 0.95
        alpha_min = 0.05

        # discount factor values
        discount_factor = 0.9
        discount_step = 0.05
        discount_max = 0.95
        discount_min = 0

        # parameters to count consecutive sign values
        m0 = 5
        m1 = 5
        accumulated_m0 = 0
        accumulated_m1 = 0

        previous_sign = "-"

        t0 = time.time()
        
        while n < NSteps:
            probabilities = softmax(H, m)
            
            k = select_arm(probabilities, m)
            
            j = 0
            g = 0
            # only uses
            while (j<nu) and ((j+n)<NSteps):
                w = scenario.generate_reward(k, n)

                sign = "+"
                if w < rewards_arm[k]/(nk[k] + 0.01):
                    sign = "-"

                # change discount factor and alpha value
                if previous_sign == sign:
                    accumulated_m1 += 1
                    accumulated_m0 = 0

                    if accumulated_m1 == m1:
                        if alpha != alpha_min:
                            alpha = round(alpha-alpha_step, 2)
                        if discount_factor != discount_min:
                            discount_factor = round(discount_factor-discount_step, 2)
                        accumulated_m1 = 0
                else:
                    accumulated_m1 = 0
                    accumulated_m0 += 1

                    if accumulated_m0 == m0:
                        if alpha != alpha_max:
                            alpha = round(alpha+alpha_step, 2)
                        if discount_factor != discount_max:
                            discount_factor = round(discount_factor+discount_step, 2)
                        accumulated_m0 = 0

                previous_sign = sign

                rewards = rewards*discount_factor
                rewards += w

                rewards_arm = rewards_arm*discount_factor
                rewards_arm[k] += w

                nk = nk*discount_factor
                nk[k] += 1

                avg_reward = rewards/sum(nk)

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

    # average the results
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