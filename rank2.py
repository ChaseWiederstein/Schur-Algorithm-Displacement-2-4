import numpy as np
from scipy.linalg import toeplitz, cholesky

def schuralg(G, R, n, debug):

    errcode = 0

    for j in range(n - 1):
        print("Round:", j)
        if debug:
            print(f"\nG_j at start of step {j + 1} of Schur Algorithm:")
            print(G)

        v = np.array([G[0, j], G[1, j]])
        print(f"Extracted {j}th vector: ", v )
        if debug:
            print(f"j-th column v of current generator:")
            print(v)

        if abs(v[0]) < abs(v[1]):
            print(f"Hyperbolic rotation does not exist for generator j = {j + 1}")
            errcode = j + 1
            return R, errcode

        tau = v[1] / v[0]
        c = 1.0 / np.sqrt(1 - tau**2)
        s = c * tau
        H = np.array([[c, -s], [-s, c]])

        G[:2, :] = H @ G[:2, :]
        if debug:
            print(f"\nG_j after hyperbolic rotation:")
            print(G)

        R[j, :] = G[0, :]

        G[0, j:n] = np.roll(G[0, j:n], 1)
        G[0, j] = 0.0

        if debug:
            print(f"\nG_j after shift right at end of step {j + 1}:")
            print(G)

    v = np.array([G[0, n - 1], G[1, n - 1]])
    if debug:
        print(f"\nn-th column v of current generator:")
        print(v)

    if abs(v[0]) < abs(v[1]):
        print(f"Hyperbolic rotation does not exist for generator j = {n}")
        errcode = n
        return R, errcode

    tau = v[1] / v[0]
    c = 1.0 / np.sqrt(1 - tau**2)
    s = c * tau
    H = np.array([[c, -s], [-s, c]])

    G[:2, :] = H @ G[:2, :]
    if debug:
        print(f"\nG_j after hyperbolic rotation:")
        print(G)

    R[n - 1, :] = G[0, :]

    return R, errcode

def driver():

    n = 10
    Id = np.eye(n)
    Z = np.hstack((Id[:, 1:], np.zeros((n, 1))))  # Downshift for displacement T - ZTZ'
    print("Z:")
    print(Z)

    ########################## TOEPLITZ EXAMPLES ######################

    # #1. Is SPD
    r = np.array([125, 60, 40, 50, 60, 20, 7, 30, 30, 20])

    #2. Is SPD. Boosts the diagonal
    # boost = 50.0
    # r = np.array([125+ boost, 50, 40, 50, 30, 20, 20, 30, 30, 20])

    #3. Sparse
    # r = np.array([100, 0, 0, 0, 10, 0, 0, 0, 5, 0])
    #
    T = toeplitz(r)
    print("T =")
    print(T)


    #################### GENERAL SPD EXAMPLES ######################
    #1. Random
    # A = np.random.randint(1, 5, size=(n, n))
    # T = np.dot(A, A.T) + 100 * np.eye(n)
    # print("T:")
    # print(T)
    # r = T[0,:]
    # print("r:")
    # print(r)

    #2. Sparse
    # A = np.random.randint(1, 5, size=(n, n))  # Generate random integers
    # for i in range(n):
    #     for j in range(n):
    #         if A[i, j] < 4:
    #             A[i, j] = 0
    #
    # T = np.dot(A, A.T)
    # T = T.astype(float) + 100 * np.eye(n)
    # T[0,0]+=20
    # print("T (Sparse SPD matrix):")
    # print(T)
    # r = T[0, :]

    # 3. Using Toeplitz multiplication
    # r = np.array([100, 0, 0, 1, 0, 0, 0, 1, 0, .01])
    # T = toeplitz(r)
    # T=(T@T.T)
    # r= T[0,:]
    # print("T:")
    # print(T)

    w = np.linalg.eigvalsh(T)
    indefv = w <= 0
    in_or_semi_count = np.sum(indefv)

    if w[0] <= 0.0:
        print("T is indefinite or semidefinite due to signs of eigenvalues.")
        print("Displacement rank 2 Schur algorithm should fail to produce a hyperbolic rotation at some step.")
        print(f"Number of eigenvalues of T <= 0: {in_or_semi_count}")

    grow1 = r / np.sqrt(r[0])
    print("grow1 =", grow1)
    grow2 = grow1 - grow1[0] * Id[:, 0]
    print("grow2 =", grow2)

    Temp = np.outer(grow1, grow1) - np.outer(grow2, grow2) # = [T1^T T1]
    print("Temp =", Temp)
    row1 = grow1.copy()
    row2 = grow2.copy()
    for _ in range(1, n):
        print("Iteration Temp: ",_)
        row1 = row1 @ Z.T
        row2 = row2 @ Z.T
        Temp += np.outer(row1, row1) - np.outer(row2, row2)
        print(Temp)


    power_series_error_norm = np.linalg.norm(Temp - T, 2) # validats T=Temp
    print(f"Matrix 2-norm of power series T and original T difference = {power_series_error_norm}")
    condition_number = np.linalg.cond(T)
    print(f"Condition number of T: {condition_number}")

    if w[0] > 0.0:
        Rtrue = cholesky(T, lower=False)
        print("Rtrue:")
        print(Rtrue)

    debug = 0

    G = np.vstack((grow1, grow2))
    print("G:")
    print(G)
    R = np.zeros((n, n))

    print("\nGenerator on input:")
    print(G)

    print("\nCalling Schur Algorithm")
    R, errcode = schuralg(G, R, n, debug)

    print("\nR on return:")
    print(R)

    if errcode == 0:
        print("Factorization completed. Check absolute Cholesky product vs original T error")
        cholesky_error = np.linalg.norm(R.T @ R - T, 2)
        print(f"Cholesky factor error: {cholesky_error}")
    else:
        print(f"Factorization failed. Error code = {errcode}")
        print(f"Hyperbolic rotation does not exist for generator j = {errcode}")
        print(f"Number of nonpositive eigenvalues count = {in_or_semi_count}")
        print(f"Smallest signed eigenvalue is lambda = {w[0]}")

if __name__ == "__main__":
    driver()
