import numpy as np
import time

# Use gaussian priors!
def ThompsonSampling2(scenario, maxG, verbose=True):
    # Get necessary parameters from the environment and simmulation
    NSteps = scenario.NSteps
    NRuns = scenario.NRuns
    m = scenario.m
    nu = scenario.nu

    # Thompson Sampling parameters
    K = 10

    # Parameters to store results
    avg_regret = np.zeros(NSteps)
    G = np.zeros((NRuns, NSteps))

    exec_time = np.zeros(NRuns)

    BA = np.zeros((NRuns, NSteps))
    avg_BA = np.zeros(NSteps)


    for i in range(NRuns):
        Q_th = np.zeros(m)
        mu_th = np.zeros(m) 
        sigma_th = np.zeros(m)
        rho_th = np.zeros(m)

        Q_dl = np.zeros(m)
        mu_dl = np.zeros(m) 
        sigma_dl = np.zeros(m)
        rho_dl = np.zeros(m)

        Q_lr = np.zeros(m)
        mu_lr = np.zeros(m) 
        sigma_lr = np.zeros(m)
        rho_lr = np.zeros(m)

        for arm in range(m):
            Q_th[arm] = np.random.random_sample()*100
            mu_th[arm] = np.random.random_sample()*100
            sigma_th[arm] = np.random.random_sample()*10

            Q_dl[arm] = np.random.random_sample()*100
            mu_dl[arm] = np.random.random_sample()*100
            sigma_dl[arm] = np.random.random_sample()

            Q_lr[arm] = np.random.random_sample()
            mu_lr[arm] = np.random.random_sample()
            sigma_lr[arm] = np.random.random_sample()

        n = 0 # initial step
        nk = np.zeros(m) # num of steps arm k was selected
        N = 0 # initial num of choices done
        Nk = np.zeros(m) # num of times arm k has been chosen

        I = np.zeros(m)

        t0 = time.time()

        # Loop for each run
        while n < NSteps:
            # Select arm according to mean estimate & uncertainty interval
            QoE = np.zeros(m)

            for arm in range(m):
                QoE[arm] = scenario.getQoE(max(Q_th[arm], 0), min(max(Q_lr[arm], 0), 1), max(Q_dl[arm], 0), n)

            k = np.argmax(QoE)
            # Get rewards for selected arm
            j = 0
            g = 0
            while (j<nu) and ((j+n)<NSteps): # always use
                th = scenario.getTH(arm, n)
                dl = scenario.getDL(arm, n)
                lr = scenario.getLR(arm, n)

                w = scenario.getQoE(th, lr, dl, n+j)

                nk[k] += 1

                old_mu_th = mu_th[k]
                mu_th[k] += 1/nk[k]*(th-mu_th[k])
                sigma_th[k] += 1/nk[k]*((th-old_mu_th)*(th-mu_th[k])-sigma_th[k])

                old_mu_dl = mu_dl[k]
                mu_dl[k] += 1/nk[k]*(dl-mu_dl[k])
                sigma_dl[k] += 1/nk[k]*((dl-old_mu_dl)*(dl-mu_dl[k])-sigma_dl[k])

                old_mu_lr = mu_lr[k]
                mu_lr[k] += 1/nk[k]*(lr-mu_lr[k])
                sigma_lr[k] += 1/nk[k]*((lr-old_mu_lr)*(lr-mu_lr[k])-sigma_lr[k])

                g += w
                j += 1

            if k == scenario.get_best_arm(n):
                BA[i, n:(n+j)] = 1

            G[i, n:(n+j)] = g + G[i, n-1]

            # Update Q-values:
            for arm in range(m):
                rho_th[arm] = sigma_th[arm]/(nk[arm]+0.01)
                rho_dl[arm] = sigma_dl[arm]/(nk[arm]+0.01)
                rho_lr[arm] = sigma_lr[arm]/(nk[arm]+0.01)

                Q_samples_th = np.zeros(K)
                Q_samples_dl = np.zeros(K)
                Q_samples_lr = np.zeros(K)
                for sample in range(K):
                    Q_samples_th[sample] = np.random.normal(mu_th[arm], rho_th[arm])
                    Q_samples_dl[sample] = np.random.normal(mu_dl[arm], rho_dl[arm])
                    Q_samples_lr[sample] = np.random.normal(mu_lr[arm], rho_lr[arm])
                
                Q_th[arm] = sum(Q_samples_th[:])/K
                Q_dl[arm] = sum(Q_samples_dl[:])/K
                Q_lr[arm] = sum(Q_samples_lr[:])/K

            n += j
            Nk[k] += 1
            N += 1


        t1 = time.time()
        exec_time[i] = t1-t0
        if verbose:
            print("Algorithm: Thompson Sampling 2. Run " + str(i+1) + "/" + str(NRuns) + " completed. Final Regret: " + str(maxG[NSteps-1]-G[i, NSteps-1]) + " Average Best-Action %: " + str(sum(BA[i, :])/NSteps) + ". Execution time: " + str(exec_time[i]) + " sec.")
    # Compute average rewards
    for i in range(NSteps):
        avg_regret[i] = maxG[i] - sum(G[:, i])/NRuns
        avg_BA[i] = sum(BA[:, i])/NRuns
    return avg_regret, avg_BA, sum(exec_time)/NRuns