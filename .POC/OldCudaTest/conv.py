import numpy as np


def conv(A, B):
    m, = A.shape
    n, = B.shape
    C = np.zeros((m + n - 1), dtype=A.dtype)
    for i in range(m):
        for j in range(n):
            C[i + j] += A[i] * B[j]
    return C


A = np.array([1, 1, 1])


B = np.array([1])
for i in range(1, 9):
    B = conv(B, A)
    print(i, list(B))
