# [2006]-"Ant Colony System"

import numpy as np
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    tau = 1  # pheromone value
    eta = 1  # heuristic desirability
    alpha = 1  # control pheromone
    beta = 1  # control heuristic
    rho = 0.2  # pheromone trail decay coefficient
    phi = 0.5  # pheromone coefficient

    if "N" in opts:
        N = opts["N"]
    if "T" in opts:
        max_iter = opts["T"]
    if "tau" in opts:
        tau = opts["tau"]
    if "alpha" in opts:
        alpha = opts["alpha"]
    if "beta" in opts:
        beta = opts["beta"]
    if "rho" in opts:
        rho = opts["rho"]
    if "eta" in opts:
        eta = opts["eta"]
    if "phi" in opts:
        phi = opts["phi"]

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]

    # Initial Tau & Eta
    tau_matrix = tau * np.ones((dim, dim))
    eta_matrix = eta * np.ones((dim, dim))

    # Pre
    fit_global = float("inf")
    fit = np.zeros(N)
    tau0 = tau_matrix.copy()

    curve = np.full(max_iter, float("inf"))
    t = 0

    # Iterations
    while t < max_iter:
        # Reset ant
        X = np.zeros((N, dim))
        for i in range(N):
            # Set number of features
            num_feat = np.random.randint(1, dim + 1)
            # Ant starts with random position
            X[i, 0] = np.random.randint(1, dim + 1)
            k = []
            if num_feat > 1:
                for d in range(1, num_feat):
                    # Start with previous tour
                    k.append(int(X[i, d - 1]))
                    print(k)
                    # Edge / Probability Selection (4)
                    P = (tau_matrix[k[-1], :] ** alpha) * (eta_matrix[k[-1], :] ** beta)
                    # Set selected position = 0 probability (4)
                    P[k] = 0
                    # Convert probability (4)
                    prob = P / P.sum()
                    # Roulette Wheel selection
                    route = roulette_wheel_selection(prob)
                    # Store selected position to be next tour
                    X[i, d] = route

        # Binary conversion
        X_bin = np.zeros((N, dim))
        for i in range(N):
            # Binary form
            ind = np.array(X[i, :], dtype=int)
            ind = ind[ind != 0]
            X_bin[i, ind - 1] = 1

        # Binary version
        for i in range(N):
            # Fitness
            fit[i] = fun(feat, label, X_bin[i, :], opts)
            # Global update
            if fit[i] < fit_global:
                Xgb = X[i, :]
                fit_global = fit[i]

        # Tau update
        tour = np.array(Xgb, dtype=int)
        tour = tour[tour != 0]
        tour = np.append(tour, tour[0])
        for d in range(len(tour) - 1):
            # Feature selected
            x = tour[d]
            y = tour[d + 1]
            # Delta tau
            Dtau = 1 / fit_global
            # Update tau (10)
            tau_matrix[x - 1, y - 1] = (1 - phi) * tau_matrix[x - 1, y - 1] + phi * Dtau

        # Evaporate pheromone (9)
        tau_matrix = (1 - rho) * tau_matrix + rho * tau0

        # Save
        curve[t] = fit_global
        print(f"\nIteration {t + 1} Best (ACS)= {curve[t]}")
        t += 1

    # Select features based on selected index
    Sf = np.unique(Xgb).tolist()
    Sf = [i - 1 for i in Sf if i != 0]
    sFeat = feat[:, Sf]

    # Store results
    acs = {
        "sf": Sf,
        "ff": sFeat,
        "nf": len(Sf),
        "c": curve,
        "f": feat,
        "l": label,
    }

    return acs


# Roulette Wheel Selection
def roulette_wheel_selection(prob):
    # Cumulative summation
    cumulative_sum = np.cumsum(prob)
    # Random one value, most probability value [0~1]
    random_val = np.random.rand()
    # Route wheel
    for i in range(len(cumulative_sum)):
        if cumulative_sum[i] > random_val:
            return i + 1
