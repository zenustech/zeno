#!/usr/bin/env python3.9

import taichi as ti
import time

ti.init(ti.cpu)

n = 8192
x = ti.field(float, (n, n))
y = ti.field(float, (n, n))

@ti.kernel
def step1():
    for i, j in x:
        y[i, j] = x[i, j - 1] + x[i, j + 1] + x[i - 1, j] + x[i + 1, j] - x[i, j] * 4


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
    return x, y;

@ti.kernel
def step2():
    bs = ti.static(16)
    for d in range((n // bs)**2):
        ib, jb = morton(d)
        for di, dj in ti.ndrange((0, bs), (0, bs)):
            i = ib + di
            j = jb + dj
            y[i, j] = x[i, j - 1] + x[i, j + 1] + x[i - 1, j] + x[i + 1, j] - x[i, j] * 4


step = step1


for i in range(3):
    step()
t0 = time.time()
for i in range(20):
    step()
y[0]
t1 = time.time()
print(t1 - t0)
