import numpy as np
import random
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    AP = 0.1  # awareness probability
    fl = 1.5  # flight length

    # Option parameters
    if "T" in opts:
        max_iter = opts["T"]
    if "N" in opts:
        N = opts["N"]
    if "AP" in opts:
        AP = opts["AP"]
    if "fl" in opts:
        fl = opts["fl"]
    if "thres" in opts:
        thres = opts["thres"]

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]

    # Initial
    X = np.zeros((N, dim))
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb + (ub - lb) * random.random()

    # Fitness
    fit = np.zeros(N)
    fitG = float("inf")
    for i in range(N):
        fit[i] = fun(feat, label, (X[i, :] > thres), opts)
        # Global update
        if fit[i] < fitG:
            fitG = fit[i]
            Xgb = X[i, :]

    # Save memory
    fitM = fit.copy()
    Xm = X.copy()
    # Pre
    Xnew = np.zeros((N, dim))

    curve = np.zeros(max_iter)
    curve[0] = fitG
    t = 2

    # Iteration
    while t <= max_iter:
        for i in range(N):
            # Randomly select one memory crow to follow
            k = np.random.randint(0, N)
            # Awareness of crow m (2)
            if random.random() >= AP:
                r = random.random()
                for d in range(dim):
                    # Crow m does not know it has been followed (1)
                    Xnew[i, d] = X[i, d] + r * fl * (Xm[k, d] - X[i, d])
            else:
                for d in range(dim):
                    # Crow m fools crow i by flying randomly
                    Xnew[i, d] = lb + (ub - lb) * random.random()

        # Fitness
        for i in range(N):
            # Fitness
            Fnew = fun(feat, label, (Xnew[i, :] > thres), opts)
            # Check feasibility
            if all(Xnew[i, :] >= lb) and all(Xnew[i, :] <= ub):
                # Update crow
                X[i, :] = Xnew[i, :]
                fit[i] = Fnew
                # Memory update (5)
                if fit[i] < fitM[i]:
                    Xm[i, :] = X[i, :]
                    fitM[i] = fit[i]
                # Global update
                if fitM[i] < fitG:
                    fitG = fitM[i]
                    Xgb = Xm[i, :]

        curve[t - 1] = fitG
        print(f"\nIteration {t} Best (CSA)= {curve[t - 1]}")
        t += 1

    # Select features
    pos = np.arange(dim)
    sf = pos[Xgb > thres]
    s_feat = feat[:, sf]

    # Store results
    csa = {"sf": sf, "ff": s_feat, "nf": len(sf), "c": curve, "f": feat, "l": label}

    return csa
