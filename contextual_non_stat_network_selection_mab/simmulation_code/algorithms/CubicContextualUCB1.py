import numpy as np
import time

def CubicContextualUCB1(scenario, maxG, verbose = True):
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
        raw_context_len = len(scenario.getContext(n))
        context_len = 0
        for pos in range(raw_context_len):
            context_len += pos
            context_len += raw_context_len - 1
        
        context_len += raw_context_len*3+1
        
        for arm in range(m):
            phi.append(np.zeros(context_len))
            theta.append(np.zeros(context_len))
            B.append(np.zeros((context_len, context_len)))
            z.append(np.zeros(context_len))

        # Initial exploration
        for arm in range(m):
            for step in range(nm):
                context = scenario.getContext(n)
                
                phi[arm] = np.zeros(context_len)
                (phi[arm])[0] = 1
                for pos in range(raw_context_len):
                    (phi[arm])[pos + 1] = context[pos]
                    (phi[arm])[pos + 1 + raw_context_len] = context[pos]**2
                    (phi[arm])[pos + 1 + 2*raw_context_len] = context[pos]**3
                pos_counter = 3*raw_context_len+1
                for i_pos in range(raw_context_len):
                    for j_pos in range(i_pos+1,raw_context_len):
                        (phi[arm])[pos_counter] = context[i_pos]*context[j_pos]
                        pos_counter += 1
                
                for i_pos in range(raw_context_len):
                    for j_pos in range(raw_context_len):
                        if i_pos != j_pos:
                            (phi[arm])[pos_counter] = (context[i_pos]**2)*context[j_pos]
                            pos_counter += 1

                
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

                phi[arm] = np.zeros(context_len)
                (phi[arm])[0] = 1
                for pos in range(raw_context_len):
                    (phi[arm])[pos + 1] = context[pos]
                    (phi[arm])[pos + 1 + raw_context_len] = context[pos]**2
                    (phi[arm])[pos + 1 + 2*raw_context_len] = context[pos]**3
                pos_counter = 3*raw_context_len+1
                for i_pos in range(raw_context_len):
                    for j_pos in range(i_pos+1,raw_context_len):
                        (phi[arm])[pos_counter] = context[i_pos]*context[j_pos]
                        pos_counter += 1
                
                for i_pos in range(raw_context_len):
                    for j_pos in range(raw_context_len):
                        if i_pos != j_pos:
                            (phi[arm])[pos_counter] = (context[i_pos]**2)*context[j_pos]
                            pos_counter += 1
                
                Q = np.dot(phi[arm], theta[arm])
                U = np.sqrt(2 * np.log(n) * np.dot(np.dot(phi[arm], B[arm]), phi[arm]))
                I[arm] = Q + U
                

            k = np.argmax(I)
            
            j = 0
            g = 0
            
            while (j < nu) and ((j+n) < NSteps):
                # Get context at step n+j
                context = scenario.getContext(n+j)
                
                phi[k] = np.zeros(context_len)
                (phi[k])[0] = 1
                for pos in range(raw_context_len):
                    (phi[k])[pos + 1] = context[pos]
                    (phi[k])[pos + 1 + raw_context_len] = context[pos]**2
                    (phi[k])[pos + 1 + 2*raw_context_len] = context[pos]**3
                pos_counter = 3*raw_context_len+1
                for i_pos in range(raw_context_len):
                    for j_pos in range(i_pos+1,raw_context_len):
                        (phi[k])[pos_counter] = context[i_pos]*context[j_pos]
                        pos_counter += 1
                
                for i_pos in range(raw_context_len):
                    for j_pos in range(raw_context_len):
                        if i_pos != j_pos:
                            (phi[k])[pos_counter] = (context[i_pos]**2)*context[j_pos]
                            pos_counter += 1

                # Get reward at step n+j
                w = scenario.generate_reward(k, n+j)

                # Update B
                tmp = np.outer(phi[k], phi[k])
                tmp = np.dot(np.dot(B[k], tmp), B[k])
                tmp = tmp/(1 + np.dot(np.dot(phi[k], B[k]), phi[k]))
                B[k] = B[k] - tmp

                # Update z
                z[k] = z[k] + w * phi[k]

                theta[k] = np.dot(B[k], z[k])
                
                g += w
                j += 1

            
            if k == scenario.get_best_arm(n):
                BA[i, n:(n+j)] = 1

            G[i, n:(n+j)] = g + G[i, n-1]

            n += j
        t1 = time.time()
        exec_time[i] = t1-t0
        if verbose:
            print("Algorithm: Cubic Contextual UCB1. Run " + str(i+1) + "/" + str(NRuns) + " completed. Final Regret: " + str(maxG[NSteps-1]-G[i, NSteps-1]) + " Average Best-Action %: " + str(sum(BA[i, :])/NSteps) + ". Execution time: " + str(exec_time[i]) + " sec.")
    # Compute average rewards
    for i in range(NSteps):
        avg_regret[i] = maxG[i] - sum(G[:, i])/NRuns
        avg_BA[i] = sum(BA[:, i])/NRuns
    return avg_regret, avg_BA, sum(exec_time)/NRuns
