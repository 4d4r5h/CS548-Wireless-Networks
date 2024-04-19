import numpy as np
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    CR = opts.get("CR", 0.8)  # crossover rate
    MR = opts.get("MR", 0.01)  # mutation rate
    tour_size = opts.get("Ts", 3)  # tournament size
    N = opts.get("N")
    max_iter = opts.get("T")

    # Objective function
    fun = Fun
    # Number of dimensions
    dim = feat.shape[1]
    # Initial
    X = initialization(N, dim)
    # Fitness
    fit = np.zeros(N)
    fitG = np.inf
    for i in range(N):
        fit[i] = fun(feat, label, X[i, :], opts)
        # Best update
        if fit[i] < fitG:
            fitG = fit[i]
            Xgb = X[i, :]

    # Pre
    curve = []
    curve.append(fitG)
    t = 1
    # Generations
    while t < max_iter:
        # Preparation
        Xc1 = np.zeros((N, dim))
        Xc2 = np.zeros((N, dim))
        fitC1 = np.zeros(N)
        fitC2 = np.zeros(N)
        z = 0
        for i in range(N):
            if np.random.rand() < CR:
                # Select two parents
                k1 = tournament_selection(fit, tour_size, N)
                k2 = tournament_selection(fit, tour_size, N)
                # Store parents
                P1 = X[k1, :]
                P2 = X[k2, :]
                # Single point crossover
                ind = np.random.randint(0, dim - 1)
                # Crossover between two parents
                Xc1[z, :] = np.concatenate((P1[: ind + 1], P2[ind + 1 :]))
                Xc2[z, :] = np.concatenate((P2[: ind + 1], P1[ind + 1 :]))
                # Mutation
                for d in range(dim):
                    # First child
                    if np.random.rand() < MR:
                        Xc1[z, d] = 1 - Xc1[z, d]
                    # Second child
                    if np.random.rand() < MR:
                        Xc2[z, d] = 1 - Xc2[z, d]
                # Fitness
                fitC1[z] = fun(feat, label, Xc1[z, :], opts)
                fitC2[z] = fun(feat, label, Xc2[z, :], opts)
                z += 1
        # Merge population
        XX = np.vstack((X, Xc1[:z], Xc2[:z]))
        FF = np.concatenate((fit, fitC1[:z], fitC2[:z]))
        # Select N best solutions
        idx = np.argsort(FF)
        X = XX[idx[:N]]
        fit = FF[idx[:N]]
        # Best agent
        if fit[0] < fitG:
            fitG = fit[0]
            Xgb = X[0]
        # Save
        curve.append(fitG)
        print(f"\nGeneration {t + 1} Best (GA Tournament)= {curve[-1]}")
        t += 1

    # Select features based on selected index
    pos = np.arange(dim)
    Sf = pos[Xgb == 1]
    sFeat = feat[:, Sf]
    # Store results
    GA = {"sf": Sf, "ff": sFeat, "nf": len(Sf), "c": curve, "f": feat, "l": label}

    return GA


def tournament_selection(fit, tour_size, N):
    # Random positions based on position and tournament size
    tour_idx = np.random.choice(N, tour_size, replace=False)
    # Select fitness value based on position selected by tournament
    tour_fit = fit[tour_idx]
    # Get position of best fitness value (win tournament)
    idx = np.argmin(tour_fit)
    # Store the position
    return tour_idx[idx]


def initialization(N, dim):
    # Initialize X vectors
    X = np.zeros((N, dim))
    for i in range(N):
        for d in range(dim):
            if np.random.rand() > 0.5:
                X[i, d] = 1
    return X
