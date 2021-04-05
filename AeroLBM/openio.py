import pyopenvdb as vdb
import numpy as np


def writevdb(path, arr):
    grid = vdb.FloatGrid()
    grid.copyFromArray(arr)
    vdb.write(path, grid)
    print('written', path)


n = 2
arr = np.float32(np.arange(n**3).reshape(n, n, n))
writevdb('/tmp/a.vdb', arr)
arr = np.ones((n, n, n))
writevdb('/tmp/b.vdb', arr)

exit(1)
