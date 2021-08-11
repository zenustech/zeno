import taichi as ti

N = 512
L = 3

mg = []
for l in range(L + 1):
    mg.append(ti.field(float, (N // 2**l, N // 2**l)))
re = []
for l in range(L + 1):
    re.append(ti.field(float, (N // 2**l, N // 2**l)))

@ti.kernel
def init():
    for i, j in mg[0]:
        mg[0][i, j] = ((i + j) % 32) / 32


@ti.kernel
def rbgs(f: ti.template(), r: ti.template(), phase: ti.template()):
    for i, j in f:
        if (i + j) % 2 != phase:
            continue
        f[i, j] = (f[i+1, j] + f[i-1, j] + f[i, j+1] + f[i, j-1] + r[i, j]) / 4

@ti.kernel
def subtract(r: ti.template(), f: ti.template()):
    for i, j in r:
        r[i, j] = 2 * (f[i+1, j] + f[i-1, j] + f[i, j+1] + f[i, j-1] + r[i, j] - 4 * f[i, j])

@ti.kernel
def restrict(xi: ti.template(), cu: ti.template()):
    for i, j in xi:
        cu[i, j] = (xi[i*2, j*2] + xi[i*2+1, j*2] + xi[i*2, j*2+1] + xi[i*2+1, j*2+1]) / 4

@ti.kernel
def prolongate(cu: ti.template(), xi: ti.template()):
    for i, j in xi:
        xi[i, j] = cu[i//2, j//2]

init()
ti.imshow(mg[0])

for l in range(L):
    rbgs(mg[l], 0)
    rbgs(mg[l], 1)
    rbgs(mg[l], 0)
    rbgs(mg[l], 1)
    subtract(re[l], mg[l])
    restrict(re[l], re[l+1])

rbgs(mg[L], 0)
rbgs(mg[L], 1)
rbgs(mg[L], 0)
rbgs(mg[L], 1)

for l in reversed(range(L)):
    prolongate(mg[l+1], mg[l])

    rbgs(mg[l], 0)
    rbgs(mg[l], 1)
    rbgs(mg[l], 0)
    rbgs(mg[l], 1)

ti.imshow(mg[0])
