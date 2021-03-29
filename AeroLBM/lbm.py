import taichi as ti
import numpy as np
import time

d2 = 1


ti.init(ti.cuda)


if d2:
#'''D2Q9
    directions_np = np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0],[0,0,0]])
    weights_np = np.array([1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,4.0/9.0])
#'''


else:
#'''D3Q15
    directions_np = np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,1,1],[-1,-1,-1],[1,1,-1],[-1,-1,1],[1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1]])
    weights_np = np.array([2.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0])
#'''

'''D3Q27
    directions_np = np.array([[0,0,0], [1,0,0],[-1,0,0],
            [0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,1,0],[-1,-1,0],[1,0,1],
            [-1,0,-1],[0,1,1],[0,-1,-1],[1,-1,0],[-1,1,0],[1,0,-1],[-1,0,1],
            [0,1,-1],[0,-1,1],[1,1,1],[-1,-1,-1],[1,1,-1],[-1,-1,1],[1,-1,1],
            [-1,1,-1],[-1,1,1],[1,-1,-1]])
    weights_np = np.array([8.0/27.0,2.0/27.0,2.0/27.0,2.0/27.0,
                2.0/27.0,2.0/27.0,2.0/27.0, 1.0/54.0,1.0/54.0,1.0/54.0
            ,1.0/54.0,1.0/54.0,1.0/54.0,1.0/54.0,1.0/54.0,1.0/54.0,
            1.0/54.0,1.0/54.0,1.0/54.0, 1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0
            ,1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0])
'''

if d2:
    res = 512, 128, 1
else:
    res = 256, 64, 64
direction_size = len(weights_np)

if d2:
    niu = 0.01
else:
    niu = 0.005
tau = 3.0 * niu + 0.5
inv_tau = 1 / tau


rho = ti.field(float, res)
vel = ti.Vector.field(3, float, res)

#f_old = ti.field(float, res + (direction_size,))
#f_new = ti.field(float, res + (direction_size,))
f_old = ti.field(float)
f_new = ti.field(float)

#ti.root.dense(ti.ijk, res).dense(ti.l, direction_size).place(f_new)
ti.root.dense(ti.ijk, 1).dense(ti.l, direction_size).dense(ti.ijk, res).place(f_new)
ti.root.dense(ti.ijk, 1).dense(ti.l, direction_size).dense(ti.ijk, res).place(f_old)

directions = ti.Vector.field(3, int, direction_size)
weights = ti.field(float, direction_size)


@ti.materialize_callback
def init_velocity_set():
    directions.from_numpy(directions_np)
    weights.from_numpy(weights_np)


@ti.kernel
def initialize():
    for x, y, z in rho:
        rho[x, y, z] = 1
        vel[x, y, z] = ti.Vector.zero(float, 3)

    for x, y, z, i in f_old:
        feq = f_eq(x, y, z, i)
        f_new[x, y, z, i] = feq
        f_old[x, y, z, i] = feq


@ti.func
def f_eq(x, y, z, i):
    eu = vel[x, y, z].dot(directions[i])
    uv = vel[x, y, z].norm_sqr()
    term = 1 + 3 * eu + 4.5 * eu**2 - 1.5 * uv
    feq = weights[i] * rho[x, y, z] * term
    return feq


@ti.kernel
def compute_density_momentum_moment():
    for x, y, z in rho:
        new_rho = 0.0
        new_vel = ti.Vector.zero(float, 3)
        for i in range(direction_size):
            f = f_new[x, y, z, i]
            f_old[x, y, z, i] = f
            new_vel += f * directions[i]
            new_rho += f
        rho[x, y, z] = new_rho
        vel[x, y, z] = new_vel / max(new_rho, 1e-6)


@ti.kernel
def collide_and_stream():
    for x, y, z in rho:
        for i in range(15):
            xmd, ymd, zmd = (ti.Vector([x, y, z]) - directions[i]) % ti.Vector(res)
            f_new[x, y, z, i] = f_old[xmd, ymd, zmd, i] * (1 - inv_tau) + f_eq(xmd, ymd, zmd, i) * inv_tau


@ti.func
def apply_bc_core(outer, bc_type, bc_vel, ibc, jbc, kbc, inb, jnb, knb):
    if (outer == 1):  # handle outer boundary
        if bc_type == 0:
            vel[ibc, jbc, kbc] = bc_vel
        elif bc_type == 1:
            vel[ibc, jbc, kbc] = vel[inb, jnb, knb]
    rho[ibc, jbc, kbc] = rho[inb, jnb, knb]
    for l in range(direction_size):
        f_old[ibc, jbc, kbc, l] = f_eq(ibc, jbc, kbc, l) - f_eq(inb, jnb, knb, l) + f_old[inb, jnb, knb, l]


@ti.kernel
def apply_bc():
    for y, z in ti.ndrange((1, res[1] - 1), (1, res[2] - 1)):
        apply_bc_core(1, 0, [0.1, 0.0, 0.0],
                0, y, z, 1, y, z)
        apply_bc_core(1, 1, [0.0, 0.0, 0.0],
                res[0] - 1, y, z, res[0] - 2, y, z)

    for x, z in ti.ndrange(res[0], res[2]):
        apply_bc_core(1, 0, [0.0, 0.0, 0.0],
                x, res[1] - 1, z, x, res[1] - 2, z)

        apply_bc_core(1, 0, [0.0, 0.0, 0.0],
                x, 0, z, x, 1, z)

    for x, y in ti.ndrange(res[0], res[1]):
        apply_bc_core(1, 0, [0.0, 0.0, 0.0],
                x, y, res[2] - 1, x, y, res[2] - 2)

        apply_bc_core(1, 0, [0.0, 0.0, 0.0], 0.0,
                x, y, 0, x, y, 1)

    for x, y, z in ti.ndrange(*res):
        pos = ti.Vector([x, y, z])
        cpos = ti.Vector(res) / ti.Vector([5, 2, 2])
        cradius = res[1] / 4
        if (pos - cpos).norm_sqr() >= cradius**2:
            continue

        vel[x, y, z] = ti.Vector.zero(float, 3)

        xnb, ynb, znb = pos + 1 if pos > cpos else pos - 1
        apply_bc_core(0, 0, [0.0, 0.0, 0.0],
                x, y, z, xnb, ynb, znb)


@ti.kernel
def apply_bc_2d():
    for y, z in ti.ndrange((1, res[1] - 1), 1):
        apply_bc_core(1, 0, [0.1, 0.0, 0.0],
                0, y, z, 1, y, z)
        apply_bc_core(1, 1, [0.0, 0.0, 0.0],
                res[0] - 1, y, z, res[0] - 2, y, z)

    for x, z in ti.ndrange(res[0], 1):
        apply_bc_core(1, 0, [0.0, 0.0, 0.0],
                x, res[1] - 1, z, x, res[1] - 2, z)

        apply_bc_core(1, 0, [0.0, 0.0, 0.0],
                x, 0, z, x, 1, z)

    for x, y, z in ti.ndrange(*res):
        pos = ti.Vector([x, y])
        cpos = ti.Vector((res[0], res[1])) / ti.Vector([5, 2])
        cradius = res[1] / 7
        if (pos - cpos).norm_sqr() >= cradius**2:
            continue

        vel[x, y, z] = ti.Vector.zero(float, 3)

        xnb, ynb = pos + 1 if pos > cpos else pos - 1
        apply_bc_core(0, 0, [0.0, 0.0, 0.0],
                x, y, z, xnb, ynb, z)


def substep():
    collide_and_stream()
    compute_density_momentum_moment()
    if d2:
        apply_bc_2d()
    else:
        apply_bc()


img = ti.field(float, (res[0], res[1]))

@ti.kernel
def render():
    for x, y in img:
        ret = 0.0
        cnt = 0
        for z in range(res[2] // 4, max(1, res[2] * 3 // 4)):
            ret += vel[x, y, z].norm() * 4
            cnt += 1
        img[x, y] = ret / cnt


#'''
initialize()
gui = ti.GUI('LBM', (1024, 256))
while gui.running and not gui.get_event(gui.ESCAPE):
    t0 = time.time()
    for subs in range(28):
        substep()
    render()
    ti.sync()
    print(time.time() - t0)
    gui.set_image(ti.imresize(img, *gui.res))
    gui.show()
'''
initialize()
for frame in range(24 * 24):

    print('compute for', frame); t0 = time.time()
    for subs in range(28):
        #print('substep', subs)
        substep()
    print('compute time', time.time() - t0)

    #grid = np.empty(res + (4,), dtype=np.float32)
    #grid[..., 3] = rho.to_numpy()
    #grid[..., :3] = vel.to_numpy()

    print('store for', frame); t0 = time.time()
    np.savez(f'/tmp/{frame:06d}', rho=rho.to_numpy(), vel=vel.to_numpy())
    print('store time', time.time() - t0)
    print('==========')
'''
