import taichi as ti
import numpy as np
import zeno
import time



ti.init(ti.cuda, kernel_profiler=True)


def vec(*args):
    return ti.Vector(args)


@ti.func
def trilerp(f: ti.template(), pos):
    p = float(pos)
    I = int(ti.floor(p))
    w0 = p - I
    w1 = 1 - w0

    c00 = f[I + vec(0,0,0)] * w1.x + f[I + vec(1,0,0)] * w0.x
    c01 = f[I + vec(0,0,1)] * w1.x + f[I + vec(1,0,1)] * w0.x
    c10 = f[I + vec(0,1,0)] * w1.x + f[I + vec(1,1,0)] * w0.x
    c11 = f[I + vec(0,1,1)] * w1.x + f[I + vec(1,1,1)] * w0.x

    c0 = c00 * w1.y + c10 * w0.y
    c1 = c01 * w1.y + c11 * w0.y

    return c0 * w1.z + c1 * w0.z


@ti.func
def trigradient(f: ti.template(), I):
    return vec(
        f[I + vec(1,0,0)] - f[I - vec(1,0,0)],
        f[I + vec(0,1,0)] - f[I - vec(0,1,0)],
        f[I + vec(0,0,1)] - f[I - vec(0,0,1)])


@ti.func
def clamp(x, xmin, xmax):
    return min(max(x, xmin), xmax)


def fieldclamped(f):
    @fieldalike
    def wrapped(*indices):
        indices = tuple(clamp(i, 0, s - 1) for i, s in zip(indices, f.shape))
        return ti.subscript(f, indices)

    return wrapped


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


class namespace:
    pass


def closureclass(base=object):
    def decorator(ctor):
        class cls(base):
            def initialize(self):
                ctor(self)

        cls.__name__ = ctor.__name__
        cls.__qualname__ = ctor.__qualname__
        cls.__doc__ = ctor.__doc__
        cls.__wrapped__ = ctor
        return cls

    return decorator


@zeno.defNodeClass
@closureclass(zeno.INode)
def LBMDomain(self):
    #res = 128, 32, 32
    res = 256, 64, 64

    rho = ti.field(float)
    vel = ti.Vector.field(3, float)
    fie = ti.field(float)

    direction_size = 15

    block = ti.root
    for _ in [rho] + vel.entries:
        block.dense(ti.ijk, res).place(_)
    block.dense(ti.indices(0, 1, 2, 3, 4), (1, 1, 1, direction_size, 2)
                ).dense(ti.ijk, res).place(fie)

    odd = ti.field(int, ())

    domain = namespace()
    domain.res = res
    domain.rho = rho
    domain.vel = vel
    domain.fie = fie
    domain.odd = odd
    domain.block = block
    self.outputs['domain'] = domain
    self.outputs['vel'] = vel

    niu = 0.02

    directions_np = np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,1,1],[-1,-1,-1],[1,1,-1],[-1,-1,1],[1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1]])
    weights_np = np.array([2.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0])

    direction_size = 15
    tau = 3.0 * niu + 0.5
    inv_tau = 1 / tau


    directions = ti.Vector.field(3, int, direction_size)
    weights = ti.field(float, direction_size)

    @ti.materialize_callback
    def init_velocity_set():
        directions.from_numpy(directions_np)
        weights.from_numpy(weights_np)


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


    @ti.materialize_callback
    @ti.kernel
    def initialize():
        for x, y, z in rho:
            rho[x, y, z] = 1
            vel[x, y, z] = ti.Vector.zero(float, 3)

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


    domain.f_cur = f_cur
    domain.f_nxt = f_nxt
    domain.direction_size = direction_size
    domain.f_eq = f_eq


    @ti.kernel
    def substep():
        for x, y, z in rho:
            if not all(0 < vec(x, y, z) < vec(*res) - 1):
                continue

            for i in range(direction_size):
                xmd, ymd, zmd = (vec(x, y, z) - directions[i]) % ti.Vector(res)
                f = f_cur[i, xmd, ymd, zmd]
                feq = f_eq(i, xmd, ymd, zmd)
                f = f * (1 - inv_tau) + feq * inv_tau
                f_nxt[i, x, y, z] = f
        swap_f_cur_nxt()

        for x, y, z in rho:
            new_rho = 0.0
            new_vel = ti.Vector.zero(float, 3)
            for i in range(direction_size):
                f = f_cur[i, x, y, z]
                new_vel += f * directions[i]
                new_rho += f
            rho[x, y, z] = new_rho
            vel[x, y, z] = new_vel / max(new_rho, 1e-6)


    self.apply = substep


@zeno.defNodeClass
@closureclass(zeno.INode)
def LBMBoundary(self):
    domain = self.inputs['domain']

    block = domain.block
    res = domain.res
    rho = domain.rho
    vel = domain.vel
    f_cur = domain.f_cur
    f_nxt = domain.f_nxt
    direction_size = domain.direction_size
    f_eq = domain.f_eq

    mask = ti.field(int)

    for _ in [mask]:
        block.dense(ti.ijk, res).place(_)


    @ti.materialize_callback
    @ti.kernel
    def initialize():
        for x, y, z in rho:
            mask[x, y, z] = 1

            pos = vec(x, y)
            cpos = vec(res[0], res[1]) / vec(5, 2)
            cradius = res[1] / 8
            if abs(pos - cpos).max() < cradius:
                mask[x, y, z] = 0

            #if 5 < x < res[0] - 5 and 5 < y < res[1] - 5 and 5 < z < res[2] - 5:
            if 1:
                #if x < 2 or res[1] - 3 < x:
                #    mask[x, y, z] = 0
                if y < 2 or res[1] - 3 < y:
                    mask[x, y, z] = 0
                if z < 2 or res[2] - 3 < z:
                    mask[x, y, z] = 0


    @ti.func
    def apply_bc_core(bc_type, bc_vel, ibc, jbc, kbc, inb, jnb, knb):
        if bc_type == 0:
            vel[ibc, jbc, kbc] = bc_vel
        elif bc_type == 1:
            vel[ibc, jbc, kbc] = vel[inb, jnb, knb]
        rho[ibc, jbc, kbc] = rho[inb, jnb, knb]
        for l in range(direction_size):
            f_cur[l, ibc, jbc, kbc] = f_eq(l, ibc, jbc, kbc) - f_eq(l, inb, jnb, knb) + f_cur[l, inb, jnb, knb]


    """
    P(x) = Q1(x)(x-3)(x-3) + 2x-5
    P(x) = Q2(x)(x-1) + 5
    P(x) = Q(x)(x-3)(x-3)(x-1) + R(x)
    P(x) = Q(x)(x-3)(x-3)(x-1) + ax + b
    P(3) = 3a + b = 2*3-5 = 1
    P(1) = a + b = 5
    a = -2
    b = 7
    """


    @ti.kernel
    def apply_bc():
        for y, z in ti.ndrange((1, res[1] - 1), (1, res[2] - 1)):
            apply_bc_core(0, [0.1, 0.0, 0.0], 0, y, z, 1, y, z)
            apply_bc_core(1, [0.0, 0.0, 0.0], res[0] - 1, y, z, res[0] - 2, y, z)

        """
        #'''
        for x, z in ti.ndrange(res[0], res[2]):
            apply_bc_core(1, 0, [0.0, 0.0, 0.0],
                    x, res[1] - 1, z, x, res[1] - 2, z)

            apply_bc_core(1, 0, [0.0, 0.0, 0.0],
                    x, 0, z, x, 1, z)
        #'''

        #'''
        for x, y in ti.ndrange(res[0], res[1]):
            apply_bc_core(1, 0, [0.0, 0.0, 0.0],
                    x, y, res[2] - 1, x, y, res[2] - 2)

            apply_bc_core(1, 0, [0.0, 0.0, 0.0],
                    x, y, 0, x, y, 1)
        #'''
        """

        for x, y, z in rho:
            if mask[x, y, z] <= 0.5:
                nrm = trigradient(fieldclamped(mask), vec(x, y, z)).normalized(1e-6)
                if nrm.norm_sqr() > 0.5:
                    vel[x, y, z] -= nrm.dot(vel[x, y, z]) * nrm
                else:
                    vel[x, y, z] = ti.Vector.zero(float, 3)


    self.apply = apply_bc

    self.outputs['mask'] = mask



@zeno.defNodeClass
@closureclass(zeno.INode)
def DyeAdvector(self):
    domain = self.inputs['domain']
    mask = self.inputs['mask']

    block = domain.block
    res = domain.res
    vel = domain.vel

    dye = ti.field(float)
    dye_nxt = ti.field(float)

    for _ in [dye, dye_nxt]:
        block.dense(ti.ijk, res).place(_)


    @ti.kernel
    def advect_dye():
        for x, y, z in dye:
            p = vec(x, y, z) - vel[x, y, z]
            p = clamp(p, 0, vec(*res) - 1)
            dye_nxt[x, y, z] = trilerp(dye, p)
        for x, y, z in dye:
            dye[x, y, z] = dye_nxt[x, y, z]
        for x, y, z in dye:
            if x < 4 and abs(y - res[1]//2) < 4 and abs(z - res[2]//2) < 4:
                dye[x, y, z] = 3
            #if mask[x, y, z] <= 0.5:
                #dye[x, y, z] = 3

    self.apply = advect_dye

    self.outputs['dye'] = dye



@zeno.defNodeClass
@closureclass(zeno.INode)
def VolumeRayMarcher(self):
    domain = self.inputs['domain']
    dye = self.inputs['dye']
    mask = self.inputs['mask']
    scale = self.params.get('scale', 1.0)
    vel = domain.vel

    res = domain.res

    img = ti.field(float, (res[0], res[1]))

    @ti.func
    def factor_at(x, y, z):
        rho = scale * dye[x, y, z]
        fac = ti.exp(-rho)
        return fac

    @ti.kernel
    def render():
        for x, y in img:
            color = 0.0
            tputz = 1.0
            for z in range(0, res[2]):
                tputy = 1.0
                for v in range(y, res[1]):
                    facy = factor_at(x, v, z)
                    tputy *= facy
                    if tputy <= 0:
                        break
                facz = factor_at(x, y, z)
                color += tputz * tputy * (1 - facz)
                tputz *= facz
                if tputz <= 0:
                    break
            img[x, y] = color

    @ti.kernel
    def render_debug():
        for x, y in img:
            color = 0.0
            count = 0
            for z in range(0, res[2]):
                #color += dye[x, y, z] * 4
                color += vel[x, y, z].norm() * 4
                count += 1
            img[x, y] = color / count

    self.apply = render

    self.outputs['img'] = img


zeno.addNode('LBMDomain', 'domain')
zeno.initNode('domain')
zeno.addNode('LBMBoundary', 'boundary')
zeno.setNodeInput('boundary', 'domain', 'domain::domain')
zeno.initNode('boundary')
zeno.addNode('DyeAdvector', 'advector')
zeno.setNodeInput('advector', 'domain', 'domain::domain')
zeno.setNodeInput('advector', 'mask', 'boundary::mask')
zeno.initNode('advector')
zeno.addNode('VolumeRayMarcher', 'render')
zeno.setNodeInput('render', 'domain', 'domain::domain')
zeno.setNodeInput('render', 'dye', 'advector::dye')
zeno.setNodeInput('render', 'mask', 'boundary::mask')
zeno.initNode('render')

gui = ti.GUI('LBM', (1024, 256))
while gui.running and not gui.get_event(gui.ESCAPE):
    t0 = time.time()
    for subs in range(28):
        zeno.applyNode('domain')
        zeno.applyNode('boundary')
        zeno.applyNode('advector')
    zeno.applyNode('render')
    img = zeno.getObject('render::img')
    gui.set_image(ti.imresize(img, *gui.res))
    print(time.time() - t0)
    gui.show()

ti.kernel_profiler_print()