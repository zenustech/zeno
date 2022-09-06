# -*- coding: utf-8 -*-
# type: ignore

import ze
import taichi as ti
print(ti)

ti.init(arch=ti.cpu)

n = 10
balls = ti.Vector.field(dtype=float, n=4, shape=(n * 2 + 1, n * 2 + 1, n * 2 + 1, n * 2 + 1))

@ti.func
def complex_mul(a, b):
    u = a[0] * b[0] - a[1] * b[1]
    v = a[1] * b[0] + a[0] * b[1]
    return ti.Vector([u, v])

@ti.func
def complex_div(a, b):
    u = a[0] * b[0] + a[1] * b[1]
    v = a[1] * b[0] - a[0] * b[1]
    return ti.Vector([u, v]) / b.norm_sqr()

@ti.func
def complex_gcd(a, b):
    # https://www.cnblogs.com/iLex/p/16222966.html
    if all(a == 0):
        a = b
    else:
        while any(b != 0):
            c = complex_div(a, b)
            q = int(c + (0.5 if c >= 0 else -0.5))
            r = a - complex_mul(b, q)
            a, b = b, r
    return a

@ti.kernel
def calc_balls():
    for a_re, a_im, b_re, b_im in balls:
        a = ti.Vector([a_re, a_im]) - n
        b = ti.Vector([b_re, b_im]) - n
        if any(b != 0) and all(complex_gcd(a, b) == ti.Vector([1, 0])):
            rad = 0.5 / b.norm_sqr()
            c = complex_div(a, b)
            if all(abs(c) <= 1):
                balls[a_re, a_im, b_re, b_im] = ti.Vector([c[0], c[1], rad, rad])

calc_balls()
balls_np = balls.to_numpy()
balls_np = balls_np[balls_np[:, :, :, :, 3] > 0]
balls_np.reshape(-1, 4)

prim = ze.ZenoPrimitiveObject.new()
prim.verts.add_attr('rad', (float, 1))
prim.verts.resize(len(balls_np))
for i, (px, py, pz, rad) in enumerate(balls_np):
    prim.verts.pos[i] = [px, py, pz]
    prim.verts.rad[i] = rad

ze.rets.obj0 = prim
