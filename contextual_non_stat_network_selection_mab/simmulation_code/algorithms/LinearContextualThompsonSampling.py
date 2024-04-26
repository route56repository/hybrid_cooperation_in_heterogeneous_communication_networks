import numpy as np
import time

def LinearContextualThompsonSampling(scenario, maxG, verbose = True):
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

    # Thompson Sampling parameters
    K = 10

    # Average over NRuns
    for i in range(NRuns):
        # Initial step
        n = 0

        Q = np.zeros(m)

        lamda = 0.5
        sigma = 0.5

        # Context vector
        phi = []


        # Other auxiliary variables
        B = []
        z = []
        mu = []
        C = []

        t0 = time.time()

        # Initialize variables
        context_len = len(scenario.getContext(n))
        for arm in range(m):
            phi.append(np.zeros(context_len))
            B.append(np.zeros((context_len, context_len)))
            z.append(np.zeros(context_len))
            mu.append(np.zeros(context_len))
            C.append(np.zeros((context_len, context_len)))

        # Initial exploration
        for arm in range(m):
            for step in range(nm):
                context = scenario.getContext(n)

                phi[arm] = np.zeros(len(context))
                for pos in range(len(context)):
                    (phi[arm])[pos] = context[pos]

                w = scenario.generate_reward(arm, n)

                # Build B matrix
                B[arm] = np.outer(phi[arm], phi[arm])

                # Build z vector
                z[arm] = w * phi[arm]

                # Build other vectors
                C[arm] = np.linalg.inv(B[arm]/(sigma**2) + np.identity((phi[arm]).size)/(lamda**2))
                mu[arm] = np.dot(C[arm]/(sigma**2), z[arm])
               

                n += 1

                

        # Loop for all the steps
        while n < NSteps:
            # Get context
            context = scenario.getContext(n)

            # Build Q vector for current context.
            for arm in range(m):
                phi[arm] = np.zeros(len(context))

                for pos in range(len(context)):
                    (phi[arm])[pos] = context[pos]

                mean = np.dot(phi[arm], mu[arm])
                var = np.dot(np.dot(phi[arm], C[arm]), phi[arm])
                
                Q_samples = np.zeros(K)

                for sample in range(K):
                    Q_samples[sample] = np.random.normal(mean, var)

                Q[arm] = sum(Q_samples[:])/K


            
            k = np.argmax(Q)

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

                # Update B
                B[k] = B[k] + np.outer(phi[k], phi[k])

                # Update z
                z[k] = z[k] + w * phi[k]

                # Update other vectors
                C[k] = np.linalg.inv(B[k]/(sigma**2) + np.identity((phi[k]).size)/(lamda**2))
                mu[k] = np.dot(C[k]/(sigma**2), z[k])

                g += w
                j += 1

            if k == scenario.get_best_arm(n):
                BA[i, n:(n+j)] = 1

            G[i, n:(n+j)] = g + G[i, n-1]

            n += j
        t1 = time.time()
        exec_time[i] = t1-t0
        if verbose:
            print("Algorithm: Linear Contextual Thompson Sampling. Run " + str(i+1) + "/" + str(NRuns) + " completed. Final Regret: " + str(maxG[NSteps-1]-G[i, NSteps-1]) + " Average Best-Action %: " + str(sum(BA[i, :])/NSteps) + ". Execution time: " + str(exec_time[i]) + " sec.")
    # Compute average rewards
    for i in range(NSteps):
        avg_regret[i] = maxG[i] - sum(G[:, i])/NRuns
        avg_BA[i] = sum(BA[:, i])/NRuns
    return avg_regret, avg_BA, sum(exec_time)/NRuns
