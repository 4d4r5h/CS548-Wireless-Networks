import numpy as np
import random
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    fmax = 2  # maximum frequency
    fmin = 0  # minimum frequency
    alpha = 0.9  # constant
    gamma = 0.9  # constant
    a_max = 2  # maximum loudness
    r0_max = 1  # maximum pulse rate

    # Option parameters
    if "N" in opts:
        N = opts["N"]
    if "T" in opts:
        max_iter = opts["T"]
    if "fmax" in opts:
        fmax = opts["fmax"]
    if "fmin" in opts:
        fmin = opts["fmin"]
    if "alpha" in opts:
        alpha = opts["alpha"]
    if "gamma" in opts:
        gamma = opts["gamma"]
    if "A" in opts:
        a_max = opts["A"]
    if "r" in opts:
        r0_max = opts["r"]
    if "thres" in opts:
        thres = opts["thres"]

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]

    # Initial
    X = np.zeros((N, dim))
    V = np.zeros((N, dim))
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb + (ub - lb) * random.random()

    # Fitness
    fit = np.zeros(N)
    fitG = float("inf")
    for i in range(N):
        fit[i] = fun(feat, label, (X[i, :] > thres), opts)
        # Global best
        if fit[i] < fitG:
            fitG = fit[i]
            Xgb = X[i, :]

    # Loudness of each bat [1 ~ 2]
    A = np.random.uniform(1, a_max, (N, 1))
    # Pulse rate of each bat [0 ~ 1]
    r0 = np.random.uniform(0, r0_max, (N, 1))
    r = r0

    # Pre
    Xnew = np.zeros((N, dim))

    curve = np.zeros(max_iter)
    curve[0] = fitG
    t = 2
    # Iterations
    while t <= max_iter:
        for i in range(N):
            # Beta [0~1]
            beta = random.random()
            # Frequency (2)
            freq = fmin + (fmax - fmin) * beta
            for d in range(dim):
                # Velocity update (3)
                V[i, d] = V[i, d] + (X[i, d] - Xgb[d]) * freq
                # Position update (4)
                Xnew[i, d] = X[i, d] + V[i, d]
            # Generate local solution around best solution
            if random.random() > r[i]:
                for d in range(dim):
                    # Epsilon in [-1,1]
                    eps = -1 + 2 * random.random()
                    # Random walk (5)
                    Xnew[i, d] = Xgb[d] + eps * np.mean(A)
            # Boundary
            Xnew[i] = np.clip(Xnew[i], lb, ub)

        # Fitness
        for i in range(N):
            # Fitness
            Fnew = fun(feat, label, (Xnew[i, :] > thres), opts)
            # Greedy selection
            if random.random() < A[i] and Fnew <= fit[i]:
                X[i, :] = Xnew[i, :]
                fit[i] = Fnew
                # Loudness update (6)
                A[i] = alpha * A[i]
                # Pulse rate update (6)
                r[i] = r0[i] * (1 - np.exp(-gamma * t))
            # Global best
            if fit[i] < fitG:
                fitG = fit[i]
                Xgb = X[i, :]

        curve[t - 1] = fitG
        print(f"\nIteration {t} Best (BA)= {curve[t - 1]}")
        t += 1

    # Select features
    pos = np.arange(dim)
    sf = pos[Xgb > thres]
    s_feat = feat[:, sf]

    # Store results
    ba = {"sf": sf, "ff": s_feat, "nf": len(sf), "c": curve, "f": feat, "l": label}

    return ba
