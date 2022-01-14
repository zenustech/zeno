#!/usr/bin/env python3.9

import taichi as ti
import time

ti.init(ti.cpu)

n = 512
x = ti.field(float, (n, n, n))
y = ti.field(float, (n, n, n))

@ti.kernel
def step1():
    for i, j, k in x:
        y[i, j, k] = x[i, j - 1, k] + x[i, j + 1, k] + x[i - 1, j, k] + x[i + 1, j, k] + x[i, j, k - 1] + x[i, j, k + 1] - x[i, j, k] * 6


@ti.func
def morton1(x):
    # https://stackoverflow.com/questions/30560214/2d-morton-decode-function-64bits
    x = x & 0x55555555;
    x = (x | (x >> 1)) & 0x33333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF;
    x = (x | (x >> 16)) & -1#0xFFFFFFFF;
    return x;

@ti.func
def morton(d):
    x = morton1(d);
    y = morton1(d >> 1);
    z = morton1(d >> 2);
    return x, y, z;

@ti.kernel
def step2():
    bs = ti.static(16)
    for d in range((n // bs)**3):
        ib, jb, kb = morton(d)
        #if ti.random() < 1e-3:
            #print(jb, jb, kb)
        for di, dj, dk in ti.ndrange((0, bs), (0, bs), (0, bs)):
            i = ib + di
            j = jb + dj
            k = kb + dk
            y[i, j, k] = x[i, j - 1, k] + x[i, j + 1, k] + x[i - 1, j, k] + x[i + 1, j, k] + x[i, j, k - 1] + x[i, j, k + 1] - x[i, j, k] * 6


step = step1


for i in range(1):
    step()
t0 = time.time()
for i in range(5):
    step()
t1 = time.time()
print(round((t1 - t0) / 5 * 1000**3), 'ns')
