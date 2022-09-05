# -*- coding: utf-8 -*-
# type: ignore

import ze
import random

prim = ze.args.obj0
prim = prim.asPrim()

n = prim.verts.size()
# prim.verts.add_attr('rad', (float, 1))
for i in range(n):
    p = prim.verts.pos[i]
    p[0] += random.random() * 0.2 - 0.1
    p[1] += random.random() * 0.2 - 0.1
    p[2] += random.random() * 0.2 - 0.1
    prim.verts.pos[i] = p
    prim.verts.rad[i] = random.random()
prim.tris.resize(0)

ze.rets.obj0 = prim
