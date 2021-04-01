import taichi as ti
import numpy as np
import time

d2 = 0


ti.init(ti.cuda, kernel_profiler=True)


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
    #res = 256, 64, 64
    res = 128, 32, 32
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
odd = ti.field(int, ())

(ti.root).dense(ti.indices(0, 1, 2, 3, 4), 1
        ).dense(ti.indices(4), direction_size
        ).dense(ti.indices(0), 2
        ).dense(ti.indices(1, 2, 3), res
        ).place(f_old)

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

    for x, y, z, i in ti.ndrange(*res, direction_size):
        feq = f_eq(x, y, z, i)
        f_old[odd[None], x, y, z, i] = feq


@ti.func
def f_eq(x, y, z, i):
    eu = vel[x, y, z].dot(directions[i])
    uv = vel[x, y, z].norm_sqr()
    term = 1 + 3 * eu + 4.5 * eu**2 - 1.5 * uv
    feq = weights[i] * rho[x, y, z] * term
    return feq


class subscripted:
    def __init__(self, func):
        self.func = func

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            indices = tuple(indices)
        elif indices is not None:
            indices = (indices,)
        else:
            indices = ()
        return self.func(*indices)


@subscripted
@ti.func
def f_cur(x, y, z, i):
    return f_old[odd[None], x, y, z, i]


@subscripted
@ti.func
def f_nxt(x, y, z, i):
    return f_old[1 - odd[None], x, y, z, i]


@ti.func
def swap_f_cur_nxt():
    odd[None] = 1 - odd[None]


@ti.kernel
def collide_stream():
    for x, y, z in rho:
        for i in range(direction_size):
            xmd, ymd, zmd = (ti.Vector([x, y, z]) - directions[i]) % ti.Vector(res)
            f_nxt[x, y, z, i] = f_cur[xmd, ymd, zmd, i] * (1 - inv_tau) + f_eq(xmd, ymd, zmd, i) * inv_tau
    swap_f_cur_nxt()
    for x, y, z in rho:
        new_rho = 0.0
        new_vel = ti.Vector.zero(float, 3)
        for i in range(direction_size):
            f = f_cur[x, y, z, i]
            new_vel += f * directions[i]
            new_rho += f
        rho[x, y, z] = new_rho
        vel[x, y, z] = new_vel / max(new_rho, 1e-6)


@ti.func
def apply_bc_core(outer, bc_type, bc_vel, ibc, jbc, kbc, inb, jnb, knb):
    if (outer == 1):  # handle outer boundary
        if bc_type == 0:
            vel[ibc, jbc, kbc] = bc_vel
        elif bc_type == 1:
            vel[ibc, jbc, kbc] = vel[inb, jnb, knb]
    rho[ibc, jbc, kbc] = rho[inb, jnb, knb]
    for l in range(direction_size):
        f_cur[ibc, jbc, kbc, l] = f_eq(ibc, jbc, kbc, l) - f_eq(inb, jnb, knb, l) + f_cur[inb, jnb, knb, l]


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

        apply_bc_core(1, 0, [0.0, 0.0, 0.0],
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
    collide_stream()
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
    continue

    print('store for', frame); t0 = time.time()
    np.savez(f'/tmp/{frame:06d}', rho=rho.to_numpy(), vel=vel.to_numpy())
    print('store time', time.time() - t0)
    print('==========')
'''
ti.kernel_profiler_print()
