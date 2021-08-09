import taichi as ti

N = 512
xi = ti.field(float, (N, N))
cu = ti.field(float, (N//2, N//2))

@ti.kernel
def init():
    for i, j in xi:
        xi[i, j] = ((i + j) % 32) / 32


@ti.kernel
def rbgs(f: ti.template(), phase: ti.template()):
    for i, j in f:
        if (i + j) % 2 != phase:
            continue
        f[i, j] = (f[i+1, j] + f[i-1, j] + f[i, j+1] + f[i, j-1]) / 4

@ti.kernel
def restrict(xi: ti.template(), cu: ti.template()):
    for i, j in xi:
        cu[i, j] = (xi[i*2, j*2] + xi[i*2+1, j*2] + xi[i*2, j*2+1] + xi[i*2+1, j*2+1]) / 4

@ti.kernel
def prolongate(cu: ti.template(), xi: ti.template()):
    for i, j in xi:
        xi[i, j] = cu[i//2, j//2]

init()
ti.imshow(xi)

rbgs(xi, 0)
rbgs(xi, 1)
rbgs(xi, 0)
rbgs(xi, 1)

restrict(xi, cu)

rbgs(cu, 0)
rbgs(cu, 1)
rbgs(cu, 0)
rbgs(cu, 1)

prolongate(cu, xi)

rbgs(xi, 0)
rbgs(xi, 1)
rbgs(xi, 0)
rbgs(xi, 1)

ti.imshow(xi)
