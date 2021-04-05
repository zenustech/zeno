import taichi as ti
ti.init(arch=ti.gpu, kernel_profiler=True)

quality = 2
n_particles = 8192 * quality**2
n_grid = 128 * quality
dx = 1 / n_grid
dt = 2e-4 / quality

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E = 400

x = ti.Vector.field(2, float)
v = ti.Vector.field(2, float)
C = ti.Matrix.field(2, 2, float)
J = ti.field(float)

grid_m = ti.field(float)
grid_v = ti.Vector.field(2, float)
pid = ti.field(int)

n_block = 4
block = ti.root.dense(ti.ij, n_grid // n_block)
block.dense(ti.ij, n_block).place(grid_v)
block.dense(ti.ij, n_block).place(grid_m)
block.dynamic(ti.k, n_block**2 * 16, n_block**2).place(pid)
ti.root.dense(ti.i, n_particles).place(x, v, C, J)

@ti.kernel
def build_pid():
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        ti.append(pid.parent(), base, p)

@ti.kernel
def do_p2g():
    #'''
    for I in ti.grouped(pid):
        p = pid[I]
    #'''
    #for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
        affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

@ti.kernel
def grid_ops():
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
        grid_v[i, j].y -= dt * gravity
        if i < bound and grid_v[i, j].x < 0:
            grid_v[i, j].x = 0
        if i > n_grid - bound and grid_v[i, j].x > 0:
            grid_v[i, j].x = 0
        if j < bound and grid_v[i, j].y < 0:
            grid_v[i, j].y = 0
        if j > n_grid - bound and grid_v[i, j].y > 0:
            grid_v[i, j].y = 0

@ti.kernel
def do_g2p():
    #'''
    for I in ti.grouped(pid):
        p = pid[I]
    #'''
    #for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        v[p] = new_v
        x[p] += dt * v[p]
        J[p] *= 1 + dt * new_C.trace()
        C[p] = new_C

def substep():
    block.deactivate_all()
    grid_m.fill(0)
    grid_v.fill(0)
    build_pid()
    do_p2g()
    grid_ops()
    do_g2p()

@ti.kernel
def init():
    for p in range(n_particles):
        x[p] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
        v[p] = [0, -1]
        J[p] = 1

init()
gui = ti.GUI('MPM88')
while gui.running and not gui.get_event(gui.ESCAPE):
    for s in range(50):
        substep()
    gui.clear(0x112F41)
    gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
    gui.show()

ti.kernel_profiler_print()