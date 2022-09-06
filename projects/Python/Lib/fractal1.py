# -*- coding: utf-8 -*-
# type: ignore

import ze
import math

balls = []

n = 64
for a in range(-n, n + 1):
    for b in range(max(1, abs(a)), n + 1):
        if math.gcd(a, b) == 1:
            rad = 0.5 / b**2
            px = a / b
            py = rad
            balls.append((px, py, 0, rad))

prim = ze.ZenoPrimitiveObject.new()
prim.verts.add_attr('rad', (float, 1))
prim.verts.resize(len(balls))
for i, (px, py, pz, rad) in enumerate(balls):
    prim.verts.pos[i] = [px, py, pz]
    prim.verts.rad[i] = rad

ze.rets.obj0 = prim
