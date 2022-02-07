#!/bin/bash

<<EOF
import sys

if 'zenblend' in sys.modules:
    sys.modules['zenblend'].unregister()

    for key in list(sys.modules.keys()):
        if key.startswith('zen'):
            del sys.modules[key]

__import__('zenblend').register()
EOF

export PYTHONPATH=`pwd`/python
export LD_LIBRARY_PATH=`pwd`/build/FastFLIP:`pwd`/build/QuickOCT:`pwd`/build/zenvdb:`pwd`/build/zenbase
if [ -z $USE_GDB ]; then
    optirun blender --python-use-system-env
else
    optirun gdb blender -ex 'r --python-use-system-env'
fi
