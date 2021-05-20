#!/bin/bash

export PYTHONPATH=`pwd`/python
export LD_LIBRARY_PATH=`pwd`/build/FastFLIP:`pwd`/build/QuickOCT:`pwd`/build/zenvdb:`pwd`/build/zenbase:`pwd`/build/MPM
if [ -z $USE_GDB ]; then
    python -m zenclient
else
    gdb python -ex 'r -m zenclient'
fi
