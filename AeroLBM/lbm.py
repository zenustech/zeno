import taichi as ti
import numpy as np
import time



ti.init(ti.cuda)


'''D2Q9
directions_np = np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0],[0,0,0]])
weights_np = np.array([1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,4.0/9.0])
'''


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

#res = 128, 32, 32
res = 256, 64, 64
direction_size = len(weights_np)

niu = 0.005
tau = 3.0 * niu + 0.5
inv_tau = 1 / tau


rho = ti.field(float)
vel = ti.Vector.field(3, float)
mask = ti.field(float)
fie = ti.field(float)


block = ti.root
for _ in [mask, rho] + vel.entries:
    block.dense(ti.ijk, res).place(_)
block.dense(ti.indices(0, 1, 2, 3, 4), (1, 1, 1, direction_size, 2)
            ).dense(ti.ijk, res).place(fie)


odd = ti.field(int, ())
directions = ti.Vector.field(3, int, direction_size)
weights = ti.field(float, direction_size)


@ti.materialize_callback
def init_velocity_set():
    directions.from_numpy(directions_np)
    weights.from_numpy(weights_np)


def vec(*args):
    return ti.Vector(args)


class fieldalike:
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

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


@fieldalike
@ti.func
def f_cur(i, x, y, z):
    return fie[x, y, z, i, odd[None]]


@fieldalike
@ti.func
def f_nxt(i, x, y, z):
    return fie[x, y, z, i, 1 - odd[None]]


@ti.func
def swap_f_cur_nxt():
    odd[None] = 1 - odd[None]


@ti.kernel
def initialize():
    for x, y, z in rho:
        rho[x, y, z] = 1
        vel[x, y, z] = ti.Vector.zero(float, 3)

        mask[x, y, z] = 1
        pos = ti.Vector([x, y, z])
        cpos = ti.Vector(res) / ti.Vector([5, 2, 2])
        cradius = res[1] / 4
        if (pos - cpos).norm_sqr() < cradius**2:
            mask[x, y, z] = 0

    for i, x, y, z in ti.ndrange(direction_size, *res):
        feq = f_eq(i, x, y, z)
        f_cur[i, x, y, z] = feq


@ti.func
def f_eq(i, x, y, z):
    eu = vel[x, y, z].dot(directions[i])
    uv = vel[x, y, z].norm_sqr()
    term = 1 + 3 * eu + 4.5 * eu**2 - 1.5 * uv
    feq = weights[i] * rho[x, y, z] * term
    return feq


@ti.kernel
def collide_stream():
    for x, y, z in rho:
        if mask[x, y, z] != 1:
            continue
        for i in range(direction_size):
            xmd, ymd, zmd = (vec(x, y, z) - directions[i]) % ti.Vector(res)
            f = f_cur[i, xmd, ymd, zmd]
            feq = f_eq(i, xmd, ymd, zmd)
            f = f * (1 - inv_tau) + feq * inv_tau
            f_nxt[i, x, y, z] = f
    swap_f_cur_nxt()


@ti.kernel
def update_macro():
    for x, y, z in rho:
        new_rho = 0.0
        new_vel = ti.Vector.zero(float, 3)
        for i in range(direction_size):
            f = f_cur[i, x, y, z]
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
        f_cur[l, ibc, jbc, kbc] = f_eq(l, ibc, jbc, kbc) - f_eq(l, inb, jnb, knb) + f_cur[l, inb, jnb, knb]


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
        if mask[x, y, z] == 2:
            vel[x, y, z] = ti.Vector.zero(float, 3)



def substep():
    collide_stream()
    update_macro()
    apply_bc()


img = ti.field(float, (res[0], res[1]))

@ti.kernel
def render():
    for x, y in img:
        ret = 0.0
        cnt = 0
        for z in range(res[2] * 3 // 8, res[2] * 5 // 8):
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
