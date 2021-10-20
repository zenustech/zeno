#!/usr/bin/blender -P

import os, sys

module_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(module_path, 'python')

if module_path not in sys.path:
    sys.path.insert(0, module_path)

print('====== restart ======')
if 'zenoblend' in sys.modules:
    sys.modules['zenoblend'].unregister()

    del sys.modules['zenoblend']
    for key in list(sys.modules.keys()):
        if key.startswith('zeno'):
            del sys.modules[key]

__import__('zenoblend').register()
