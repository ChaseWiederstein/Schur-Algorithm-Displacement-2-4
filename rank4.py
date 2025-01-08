import numpy as np
from scipy.linalg import toeplitz, cholesky

def rank4_schuralg(G, R, n, debug):

    errcode = 0

    for j in range(n - 1):
        if debug:
            print(f"\nG_j at start of step {j + 1} of Schur Algorithm:")
            print(G)

        for block in range(0, 4, 2):
            v = G[block:block + 2, j]
            murot = np.abs(v[0]) + np.abs(v[1])
            if murot < 1e-9:
                c, s = 1.0, 0.0
            else:
                rho = murot * np.sqrt((v[0] / murot)**2 + (v[1] / murot)**2)
                c, s = v[0] / rho, v[1] / rho
            Prot = np.array([[c, s], [-s, c]])
            G[block:block + 2, :] = Prot @ G[block:block + 2, :]
            if debug:
                print(f"\n{block + 1},{block + 2} plane rotation for G_j")
                print(Prot)
                print(G)

        v = np.array([G[0, j], G[2, j]])
        if abs(v[0]) < abs(v[1]):
            print(f"Hyperbolic rotation does not exist for generator j = {j + 1}")
            errcode = j + 1
            return R, errcode

        tau = v[1] / v[0]
        c = 1.0 / np.sqrt(1 - tau**2)
        s = c * tau
        H = np.array([[c, -s], [-s, c]])
        G[[0, 2], :] = H @ G[[0, 2], :]
        if debug:
            print(f"\nG_j after hyperbolic rotation:")
            print(G)

        # update r with the current row g
        R[j, :] = G[0, :]

        # shift right
        G[0, j:n] = np.roll(G[0, j:n], 1)
        G[0, j] = 0.0
        if debug:
            print(f"\nG_j after shift right at end of step {j + 1}:")
            print(G)

    # last col
    v = np.array([G[0, n - 1], G[2, n - 1]])
    if abs(v[0]) < abs(v[1]):
        print(f"Hyperbolic rotation does not exist for generator j = {n}")
        errcode = n
        return R, errcode

    tau = v[1] / v[0]
    c = 1.0 / np.sqrt(1 - tau**2)
    s = c * tau
    H = np.array([[c, -s], [-s, c]])
    G[[0, 2], :] = H @ G[[0, 2], :]
    if debug:
        print(f"\nG_j after hyperbolic rotation:")
        print(G)

    R[n - 1, :] = G[0, :]

    return R, errcode

def driver():

    m, n = 5, 5
    #1. square dense
    c = np.array([12, 4, 3, 2, 1])
    r = np.array([12, 4, 3, 2, 1])

    #2. Square dense boosted
    # c = np.array([100, 4, 3, 2, 1])
    # r = np.array([100, 4, 3, 2, 1])
    #
    # #3. Sparse
    # c = np.array([5, 1, 0, 0, 0])
    # r = np.array([5, 0, 0, 0, 0])

    # c = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    # r = np.array([10, 3, 4, 5, 6])
    T = toeplitz(c[:m], r[:n])
    print("T:")
    print(T)

    N = T.T @ T

    b = np.random.randn(m)
    d = T.T @ b

    Rtrue = cholesky(N, lower=False)
    print("\nTrue Cholesky factor of N:")
    print(Rtrue)

    Id = np.eye(n)
    Z = np.hstack((Id[:, 1:], np.zeros((n, 1))))
    NabN = N - Z @ N @ Z.T

    # compoute generator
    alpha = np.linalg.norm(T[:, 0], 2)
    w1 = T.T @ T[:, 0] / alpha
    w2 = T[0, :].copy()
    w2[0] = 0.0
    w3 = np.roll(w1, -1)
    w3[-1] = 0.0
    w4 = np.concatenate(([0.0], T[-1, :-1][::-1]))

    G = np.column_stack((w1, w2, w3, w4))
    print("\nGenerator G:")
    print(G)

    Phi = np.block([
        [np.eye(2), np.zeros((2, 2))],
        [np.zeros((2, 2)), -np.eye(2)]
    ])

    NabN_comp = G @ Phi @ G.T

    N_comp = NabN_comp.copy()
    Gshifted = G.copy()
    for _ in range(1, n):
        Gshifted = Z @ Gshifted
        N_comp += Gshifted @ Phi @ Gshifted.T

    NabErr = np.linalg.norm(NabN - NabN_comp, 2)
    Nerror = np.linalg.norm(N - N_comp, 2)
    print(f"Displacement error = {NabErr}, Normal Equation Matrix error = {Nerror}\n")
    kappa_N = np.linalg.cond(N)
    print(f"Condition number of N: {kappa_N}")

    print("Calling rank-4 Schur Algorithm")
    R = np.zeros((n, n))
    debug = 0
    R, errcode = rank4_schuralg(G.T, R, n, debug)

    print("\nR on return:")
    print(R)

    if errcode == 0:
        Rerror = np.linalg.norm(Rtrue - R, 2)
        print(f"Cholesky factor error = {Rerror}")
    else:
        print(f"Factorization failed. Error code = {errcode}")

if __name__ == "__main__":
    driver()
