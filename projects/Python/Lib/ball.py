# -*- coding: utf-8 -*-
# type: ignore

import ze

balls = []

n = 16
def complex_gcd(a, b):
    # https://www.cnblogs.com/iLex/p/16222966.html
    if a == 0: return b
    while b != 0:
        u = a.real * b.real + a.imag * b.imag
        v = b.real * a.imag - a.imag * b.real
        b2 = abs(b)**2
        q = complex(round(u / b2), round(v / b2))
        if q == 0:
            break
        r = a - b * q
        a, b = b, r
    return a

for a_re in range(-n, n + 1):
    for a_im in range(-n, n + 1):
        for b_re in range(-n, n + 1):
            for b_im in range(-n, n + 1):
                a = complex(a_re, a_im)
                b = complex(b_re, b_im)
                if b != 0 and complex_gcd(a, b) == 1:
                    rad = 0.5 / abs(b)**2
                    c = a / b
                    if abs(c.real) <= 1 and abs(c.imag) <= 1:
                        px = c.real
                        py = c.imag
                        pz = rad
                        balls.append((px, py, pz, rad))

print(len(balls))

prim = ze.ZenoPrimitiveObject.new()
prim.verts.add_attr('rad', (float, 1))
prim.verts.resize(len(balls))
for i, (px, py, pz, rad) in enumerate(balls):
    prim.verts.pos[i] = [px, py, pz]
    prim.verts.rad[i] = rad

ze.rets.obj0 = prim
