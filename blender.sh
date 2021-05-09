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
export LD_LIBRARY_PATH=`pwd`/build/FastFLIP:`pwd`/build/QuickOCT
optirun blender --python-use-system-env
