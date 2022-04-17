#!/usr/bin/env python

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))

import math
from zenapi.zpmio import writezpm

with open(sys.argv[1], 'r') as fin:
    lines = fin.readlines()

mass = []
pos = []
vel = []
for line in lines:
    try:
        m, x, y, z, u, v, w = map(float, line.split())
    except ValueError:
        pass
    mass.append(m)
    pos.append((x, y, z))
    vel.append((u, v, w))
writezpm(sys.argv[2], dict(mass=mass, pos=pos, vel=vel))
