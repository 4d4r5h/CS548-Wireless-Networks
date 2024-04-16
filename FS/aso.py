import numpy as np
from FS.functionHO import Fun


def jfs(feat, label, opts):
    # Parameters
    lb = 0
    ub = 1
    thres = 0.5
    alpha = 50  # depth weight
    beta = 0.2  # multiplier weight

    if "N" in opts:
        N = opts["N"]
    if "T" in opts:
        max_Iter = opts["T"]
    if "alpha" in opts:
        alpha = opts["alpha"]
    if "beta" in opts:
        beta = opts["beta"]
    if "thres" in opts:
        thres = opts["thres"]

    # Objective function
    fun = Fun

    # Number of dimensions
    dim = feat.shape[1]

    # Initial
    X = np.random.uniform(lb, ub, (N, dim))
    V = np.random.uniform(lb, ub, (N, dim))

    # Pre
    temp_A = np.zeros((N, dim))
    fitG = float("inf")
    fit = np.zeros(N)

    curve = float("inf")
    t = 1

    # Iteration
    while t <= max_Iter:
        for i in range(N):
            # Fitness
            fit[i] = fun(feat, label, (X[i, :] > thres), opts)
            # Best update
            if fit[i] < fitG:
                fitG = fit[i]
                Xbest = X[i, :]

        # Best & worst fitness
        fitB = np.min(fit)
        fitW = np.max(fit)
        # Number of K neighbor
        Kbest = np.ceil(N - (N - 2) * np.sqrt(t / max_Iter)).astype(int)
        # Mass
        M = np.exp(-(fit - fitB) / (fitW - fitB))
        # Normalized mass
        M = M / np.sum(M)
        # Sort normalized mass in descending order
        idx_M = np.argsort(-M)
        # Constraint force
        G = np.exp(-20 * t / max_Iter)
        E = np.zeros((N, dim))

        for i in range(N):
            XK = np.sum(X[idx_M[:Kbest], :], axis=0) / Kbest
            # Length scale
            scale_dist = np.linalg.norm(X[i, :] - XK)
            for ii in range(Kbest):
                # Select neighbor with higher mass
                j = idx_M[ii]
                # Get LJ-potential
                Po = jLJPotential(X[i, :], X[j, :], t, max_Iter, scale_dist)
                # Distance
                dist = np.linalg.norm(X[i, :] - X[j, :])
                for d in range(dim):
                    # Update
                    E[i, d] += (
                        np.random.random()
                        * Po
                        * ((X[j, d] - X[i, d]) / (dist + np.finfo(float).eps))
                    )

            for d in range(dim):
                E[i, d] = alpha * E[i, d] + beta * (Xbest[d] - X[i, d])
                # Calculate part of acceleration
                temp_A[i, d] = E[i, d] / M[i]

        # Update
        for i in range(N):
            for d in range(dim):
                # Acceleration
                Acce = temp_A[i, d] * G
                # Velocity update
                V[i, d] = np.random.random() * V[i, d] + Acce
                # Position update
                X[i, d] = X[i, d] + V[i, d]
            # Boundary
            X[i, :] = np.clip(X[i, :], lb, ub)

        curve[t - 1] = fitG
        print(f"Iteration {t} Best (ASO) = {curve[t - 1]}")
        t += 1

    # Select features based on selected index
    Pos = np.arange(dim)
    Sf = Pos[Xbest > thres]
    sFeat = feat[:, Sf]

    # Store results
    ASO = {"sf": Sf, "ff": sFeat, "nf": len(Sf), "c": curve, "f": feat, "l": label}

    return ASO


def jLJPotential(X1, X2, t, max_Iter, scale_dist):
    # Calculate LJ-potential
    h0 = 1.1
    u = 1.24
    # Equilibration distance [Assume 1.12*(17)~=(17)]
    r = np.linalg.norm(X1 - X2)
    # Depth function
    n = (1 - (t - 1) / max_Iter) ** 3
    # Drift factor
    g = 0.1 * np.sin((np.pi / 2) * (t / max_Iter))
    # Hmax & Hmin
    Hmin = h0 + g
    Hmax = u
    # Compute H
    if r / scale_dist < Hmin:
        H = Hmin
    elif r / scale_dist > Hmax:
        H = Hmax
    else:
        H = r / scale_dist

    # Revised version
    Potential = n * (12 * (-H) ** (-13) - 6 * (-H) ** (-7))
    return Potential
