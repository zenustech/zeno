import os

from zenutils import load_library, rel2abs

modsdir = rel2abs(__file__, 'mods')

for name in os.listdir(modsdir):
    if not name.endswith('.so'): continue
    print('loading addon:', name)
    load_library(os.path.join(modsdir, name))

__all__ = []
