#!/usr/bin/env python

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))

from zenapi.zpmio import writezpm

skip = int(sys.argv[1])

with open('galaxy/data/dubinski.tab', 'r') as fin:
    lines = fin.readlines()
    lines = lines[::skip]
    lines = lines[:len(lines) // 4 * 4]

mass = []
pos = []
vel = []
for line in lines:
    m, x, y, z, u, v, w = map(float, line.split())
    mass.append(m)
    pos.append((x, y, z))
    vel.append((u, v, w))
writezpm('dubinski.zpm', dict(mass=mass, pos=pos, vel=vel))
