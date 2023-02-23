# How to cihou Zeno-Python

1. Uninstall Wendous!

2. Turn on python project and build:
```bash
cmake -B build -DZENO_WITH_Python:BOOL=ON
cmake --build build
```

3. Install numpy locally via pip:
```bash
build/bin/zenoedit -invoke python -m ensurepip
build/bin/zenoedit -invoke python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ numpy
```

4. Play with `PythonScript` node with `code` being multiline string:
```python
import ze
import numpy as np
prim = ze.ZenoPrimitiveObject.new()
prim.verts.resize(4)
pos_np = np.array([
 [0, 0, 0],
 [0, 0, 1],
 [0, 1, 0],
 [1, 0, 0],
])
prim.verts.pos.from_numpy(pos_np)
ze.rets.prim = prim
```

5. Use `ExtractDict` to get `rets` output key `prim`, use a `Route` to VIEW it.

See `misc/graphs/cihounumpytest.zsg` for full demo.
