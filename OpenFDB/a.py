import sys
sys.path.append('..')
from zenqt.system.zpmio import writezpm

pos = []
sdf = []
with open('a.txt', 'r') as f:
    for line in f.readlines():
        x, y, z, w = map(float, line.split())
        #print(x, y, z, w)
        pos.append([x, y, z])
        sdf.append(w)

writezpm('/tmp/a.zpm', {'pos': pos, 'sdf': sdf})
