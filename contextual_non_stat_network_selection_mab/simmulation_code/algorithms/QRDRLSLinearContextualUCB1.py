import numpy as np
import time

def QRDRLSLinearContextualUCB1(scenario, maxG, verbose = True):
    # Get necessary parameters from the environment  and simmulation
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

    discount_factor = 0.9995 #0.999

    # Average over NRuns
    for i in range(NRuns):
        # Initial step
        n = 0

        I = np.zeros(m)

        _lambda = 0.1

        # Context vector
        phi = []

        # Theta vector
        theta = []

        # Other auxiliary variables
        R = []
        z = []
        A = [] # only used for first L steps for a specific arm
        B = []
        r = [] # only used for first L steps for a specific arm
        t0 = time.time()
        
        # Initialize variables
        context_len = len(scenario.getContext(n))
        for arm in range(m):
            phi.append(np.zeros(context_len))
            theta.append(np.zeros(context_len))

            A.append(np.zeros(context_len))
            B.append(np.zeros((context_len, context_len)))

            r.append(np.zeros(context_len))

            # QR decomposition matrices
            R.append(np.zeros((context_len, context_len)))
            z.append(np.zeros(context_len))

        # Initial exploration
        for arm in range(m):
            for step in range(nm):
                context = scenario.getContext(n)

                phi[arm] = np.zeros(context_len)
                for pos in range(context_len):
                    (phi[arm])[pos] = context[pos]

                w = scenario.generate_reward(arm, n)

                # Initialize A matrix
                A[arm] = np.zeros((1, context_len))
                for x_step in range(context_len):
                    (A[arm])[0][x_step] = (phi[arm])[x_step]

                (r[arm])[0] = w

                # Initialize B matrix
                B[arm] = np.linalg.inv(_lambda * np.identity(context_len) + np.outer(phi[arm], phi[arm]))

                # Build z vector
                z[arm] = w * phi[arm]

                # Build theta vector
                theta[arm] = np.dot(B[arm], z[arm])

                n += 1

        # Loop for all the steps
        while n < NSteps:
            # Get context
            context = scenario.getContext(n)

            for arm in range(m):
                phi[arm] = np.zeros(context_len)

                for pos in range(context_len):
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

                x_A, y_A = A[k].shape
                if x_A >= context_len:
                    # Use QR Update
                    # Update R, z
                    A_matrix = np.zeros((context_len+1, context_len+1))

                    for x_step in range(context_len):
                        for y_step in range(context_len):
                            A_matrix[x_step][y_step] = discount_factor*(R[k])[x_step][y_step]

                    for x_step in range(context_len):
                        A_matrix[x_step][context_len] = discount_factor*(z[k])[x_step]

                    for y_step in range(context_len):
                        A_matrix[context_len][y_step] = (phi[k])[y_step]

                    A_matrix[context_len][context_len] = w

                    Q, R_matrix = np.linalg.qr(A_matrix, mode = 'reduced')

                    for x_step in range(context_len):
                        for y_step in range(context_len):
                            (R[k])[x_step][y_step] = R_matrix[x_step][y_step]

                    for x_step in range(context_len):
                        (z[k])[x_step] = R_matrix[x_step][context_len]


                    theta[k] = np.dot(np.linalg.inv(R[k]), z[k])
                    B[k] = np.linalg.inv(np.dot(R[k].transpose(), R[k])+ 0.01* np.identity(context_len))

                else:
                    # Update A
                    A_old = A[k]
                    A[k] = np.zeros((x_A+1, y_A))
                    for x_step in range(x_A):
                        for y_step in range(y_A):
                            (A[k])[x_step][y_step] = A_old[x_step][y_step]

                    for y_step in range(context_len):
                        (A[k])[x_A][y_step] = (phi[k])[y_step]

                    (r[k])[x_A] = w
                    
                    # Use usual contextual math
                    # Update B
                    tmp = np.outer(phi[k], phi[k])
                    tmp = np.dot(np.dot(B[k], tmp), B[k])
                    tmp = tmp/(1 + np.dot(np.dot(phi[k], B[k]), phi[k]))
                    B[k] = B[k] - tmp
                    
                    # Update z
                    z[k] = z[k] + w * phi[k]

                    # Update theta
                    theta[k] = np.dot(B[k], z[k])

                    # Check if we can start QRD
                    x_A, y_A = A[k].shape
                    if x_A == context_len:
                        Q, R[k] = np.linalg.qr(A[k], mode = 'reduced')
                        tmp = np.dot(Q, R[k])
                        
                        z[k] = np.dot(Q.transpose(), r[k]) 

                        R[k] = R[k] + 0.01* np.identity(context_len)
                        theta[k] = np.dot(np.linalg.inv(R[k]), z[k])
                        B[k] = np.linalg.inv(np.dot(R[k].transpose(), R[k]))

                g += w
                j += 1
            if k == scenario.get_best_arm(n):
                BA[i, n:(n+j)] = 1

            G[i, n:(n+j)] = g + G[i, n-1]

            n += j

        t1 = time.time()
        exec_time[i] = t1 - t0
        if verbose:
            print("Algorithm: QRD RLS Linear Contextual UCB1. Run " + str(i+1) + "/" + str(NRuns) + " completed. Final Regret: " + str(maxG[NSteps-1]-G[i, NSteps-1]) + " Average Best-Action %: " + str(sum(BA[i, :])/NSteps) + ". Execution time: " + str(exec_time[i]) + " sec.")
    # Compute average rewards
    for i in range(NSteps):
        avg_regret[i] = maxG[i] - sum(G[:, i])/NRuns
        avg_BA[i] = sum(BA[:, i])/NRuns
    return avg_regret, avg_BA, sum(exec_time)/NRuns