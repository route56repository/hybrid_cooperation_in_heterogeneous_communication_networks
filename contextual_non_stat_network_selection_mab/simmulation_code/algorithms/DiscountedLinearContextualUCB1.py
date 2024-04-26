import numpy as np
import time

def DiscountedLinearContextualUCB1(scenario, maxG, verbose = True):
    # Get necessary parameters from the environment and simmulation
    NSteps = scenario.NSteps
    NRuns = scenario.NRuns
    m = scenario.m
    nu = scenario.nu
    nm = scenario.nm

    # Parameters to store results
    avg_regret = np.zeros(NSteps)
    G = np.zeros((NRuns, NSteps))

    exec_time = np.zeros(NRuns)

    BA = np.zeros((NRuns, NSteps))
    avg_BA = np.zeros(NSteps)

    # Discounted constants
    discount_factor = 0.999994

    # Average over NRuns
    for i in range(NRuns):
        mu = np.zeros(m)

        # Initial Step
        n = 0

        I = np.zeros(m)

        lamda = 0.1

        # Context vector
        phi = []

        # Theta vector ('mean vector')
        theta = []

        # Other auxiliary variables
        B = []
        z = []

        t0 = time.time()

        # Initialize variables
        context_len = len(scenario.getContext(n))
        for arm in range(m):
            phi.append(np.zeros(context_len))
            theta.append(np.zeros(context_len))
            B.append(np.zeros((context_len, context_len)))
            z.append(np.zeros(context_len))

        # Initial exploration
        for arm in range(m):
            for step in range(nm):
                context = scenario.getContext(n)

                phi[arm] = np.zeros(len(context))
                for pos in range(len(context)):
                    (phi[arm])[pos] = context[pos]

                w = scenario.generate_reward(arm, n)

                # Build B matrix
                B[arm] = np.linalg.inv(lamda * np.identity((phi[arm]).size) + np.outer(phi[arm], phi[arm]))

                # Build z vector
                z[arm] = w * phi[arm]

                # Build theta vector
                theta[arm] = np.dot(B[arm], z[arm])

                n += 1

        # Loop for all the steps
        while n < NSteps:
            # Get context
            context = scenario.getContext(n)

            # Build I = Q + U for current context
            for arm in range(m):
                phi[arm] = np.zeros(len(context))

                for pos in range(len(context)):
                    (phi[arm])[pos] = context[pos]

                Q = np.dot(phi[arm], theta[arm])
                U = np.sqrt(2 * np.log(n) * np.dot(np.dot(phi[arm], B[arm]), phi[arm]))
                I[arm] = Q + U
                

            k = np.argmax(I)
            
            j = 0
            g = 0
            
            while (j < nu) and ((j+n) < NSteps):
                # Get context at step n+j
                context = scenario.getContext(n+j)
                phi[k] = np.zeros(len(context))
                for pos in range(len(context)):
                    (phi[k])[pos] = context[pos]

                # Get reward at step n+j
                w = scenario.generate_reward(k, n+j)

                # Update theta
                tmp = np.dot(np.dot(phi[k], B[k]), z[k])
                mu = tmp/(1 + np.dot(np.dot(phi[k], B[k]), phi[k]))

                tmp = np.dot(np.dot(phi[k], B[k]), phi[k]) * w
                gamma = tmp/(1 + np.dot(np.dot(phi[k], B[k]), phi[k]))

                theta[k] = theta[k] - np.dot(B[k], phi[k]) * (mu + gamma)

                # Update B
                tmp = np.outer(phi[k], phi[k])
                tmp = np.dot(np.dot(B[k], tmp), B[k])
                tmp = tmp/(1 + np.dot(np.dot(phi[k], B[k]), phi[k]))
                B[k] = B[k] - tmp

                # Update z
                z[k] = discount_factor*z[k] + w * phi[k]

                theta[k] = np.dot(B[k], z[k])

                for arm in range(m):
                    if arm != k:
                        z[arm] = discount_factor*z[arm]
                        theta[arm] = np.dot(B[arm], z[arm])

                
                g += w
                j += 1
                
            if k == scenario.get_best_arm(n):
                BA[i, n:(n+j)] = 1

            G[i, n:(n+j)] = g + G[i, n-1]

            n += j
        t1 = time.time()
        exec_time[i] = t1-t0
        if verbose:
            print("Algorithm: Discounted Linear Contextual UCB1. Run " + str(i+1) + "/" + str(NRuns) + " completed. Final Regret: " + str(maxG[NSteps-1]-G[i, NSteps-1]) + " Average Best-Action %: " + str(sum(BA[i, :])/NSteps) + ". Execution time: " + str(exec_time[i]) + " sec.")
    # Compute average rewards
    for i in range(NSteps):
        avg_regret[i] = maxG[i] - sum(G[:, i])/NRuns
        avg_BA[i] = sum(BA[:, i])/NRuns
    return avg_regret, avg_BA, sum(exec_time)/NRuns
