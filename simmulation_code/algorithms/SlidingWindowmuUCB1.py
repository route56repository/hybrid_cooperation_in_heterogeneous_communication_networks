import numpy as np
import time

# muUCB1 algorithm with Sliding Window variation
def SlidingWindowmuUCB1(scenario, maxG):
    # Get necessary parameters from the environment and simmulation
    NSteps = scenario.NSteps
    NRuns = scenario.NRuns
    m = scenario.m
    nu = scenario.nu
    nm = scenario.nm

    # Sliding Window Constants
    window_size = 5000

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

        selected_arms = np.zeros(NSteps)
        rewards = np.zeros(NSteps)
        rewards_window = np.zeros(m)

        actions = []

        I = np.zeros(m)

        t0 = time.time()
        
        while n < NSteps:
            I = mu + np.sqrt((2*np.log(max(N, 1)))/(Nk+0.01))
            k = np.argmax(I) 
            
            # use during nu steps
            if k == np.argmax(mu):
                j = 0
                g = 0
                while (j<nu) and ((j+n)<NSteps):
                    w = scenario.generate_reward(k, n)

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

                # Update estimate values
                for arm in range(m):
                    mu[arm] = rewards_window[arm]/max(nk[arm], 0.01)

                if k == scenario.get_best_arm(n):
                    BA[i, n:(n+j)] = 1

                G[i, n:(n+j)] = g + G[i, n-1]

                n += j

                actions.append([k, "nu"])
                Nk[k] += 1

                if n > window_size:
                    action = actions.pop()
                    counter = 0
                    while action[1] != "nu" and counter < nu:
                        Nk[action[0]] -= 1
                        action = actions.pop(0)
                        counter += nm

                N = sum(Nk)

            # measure during nm steps
            else:
                j = 0
                while (j<nm) and ((j+n)<NSteps):
                    w = scenario.generate_reward(k, n)

                    selected_arms[n+j] = k
                    nk[k] += 1
                    if (n+j) >= window_size:
                        nk[int(selected_arms[n+j-window_size])] -= 1

                    rewards[n+j] = w
                    rewards_window[k] += w
                    if (n+j) >= window_size:
                        rewards_window[int(selected_arms[n+j-window_size])] -= rewards[n+j-window_size]

                    j += 1

                # Update estimate values
                for arm in range(m):
                    mu[arm] = rewards_window[arm]/max(nk[arm], 0.01)

                G[i, n:(n+j)] = G[i, n-1]

                n += j

                actions.append([k, "nm"])

                Nk[k] += 1
                if n > window_size:
                    action = actions.pop()
                    Nk[action[0]] -= 1

                N = sum(Nk)
            
            
        t1 = time.time()
        exec_time[i] = t1-t0

    # Compute average rewards
    for i in range(NSteps):
        avg_regret[i] = maxG[i] - sum(G[:, i])/NRuns
        avg_BA[i] = sum(BA[:, i])/NRuns
    return avg_regret, avg_BA, sum(exec_time)/NRuns