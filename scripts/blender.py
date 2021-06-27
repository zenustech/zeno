# run these in blender scripting module:

modpath = '/home/bate/Codes/zeno'
pkgname = 'zenblend'
pkgpattern = 'zen'

import sys

if modpath not in sys.path:
    sys.path.insert(0, modpath)

if pkgname in sys.modules:
    sys.modules[pkgname].unregister()

    del sys.modules[pkgname]
    keys = list(sys.modules.keys())
    for key in keys:
        if key.startswith(pkgpattern):
            del sys.modules[key]

__import__(pkgname).register()
