# https://www.wias-berlin.de/people/john/LEHRE/MULTIGRID/multigrid_6.pdf
import numpy as np
import time



def smooth(v, f, times):
    n = len(v)
    for t in range(times):
        w = np.zeros(n)
        w[0] = 0.5 * (v[1] + f[0])
        w[n - 1] = 0.5 * (v[n - 2] + f[n - 1])
        for i in range(1, n - 1):
            w[i] = 0.5 * (v[i - 1] + v[i + 1] + f[i])
        v = w
    return v


def residual(v, f):
    n = len(v)
    r = np.zeros(n)
    r[0] = v[1] - 2 * v[0] + f[0]
    r[n - 1] = v[n - 2] - 2 * v[n - 1] + f[n - 1]
    for i in range(1, n - 1):
        r[i] = v[i - 1] + v[i + 1] - 2 * v[i] + f[i]
    return r


def restrict(v):
    n = len(v)
    m = n // 2
    w = np.zeros(m)
    for i in range(m):
        w[i] = 0.5 * (v[i * 2] + v[i * 2 + 1])
    return w


def prolongate(v):
    n = len(v)
    m = n * 2
    w = np.zeros(m)
    om = 0.2
    for i in range(n - 1):
        w[i * 2] = (1 - om) * v[i] + om * v[i + 1]
        w[i * 2 + 1] = om * v[i] + (1 - om) * v[i + 1]
    return w



N = 2**18
f = np.zeros(N)
f[N // 2] = 32
v = np.zeros(N)


def mgsolvev1(vh, fh):
    vh = smooth(vh, fh)
    r2h = restrict(residual(vh, fh))
    e2h = smooth(r2h, r2h)

    eh = prolongate(e2h)
    vh = vh + eh
    vh = smooth(vh, fh)

    return vh


def mgsolvev2(v, f, levels=None):
    if levels is None:
        levels = max(1, int(np.log(len(v))) - 10)

    v = [v] + [None] * levels
    f = [f] + [None] * levels

    v[0] = smooth(v[0], f[0], 4)

    for i in range(levels):
        f[i + 1] = restrict(residual(v[i], f[i]))
        v[i + 1] = smooth(f[i + 1] * 0, f[i + 1], 4 << i)

    for i in reversed(range(levels)):
        v[i] = smooth(v[i] + prolongate(v[i + 1]), f[i], 4 << i)

    return v[0]


def norm(x):
    return np.linalg.norm(x), max(np.abs(x))



'''
t0 = time.time()
r = residual(smooth(v, f, 8), f)
print(*norm(r), time.time() - t0)

t0 = time.time()
r = residual(mgsolvev1(v, f), f)
print(*norm(r), time.time() - t0)
'''

t0 = time.time()
r = residual(mgsolvev2(v, f), f)
print(*norm(r), time.time() - t0)
