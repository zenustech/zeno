#!/bin/bash

<<<<<<< HEAD
export PYTHONPATH=`pwd`/python
export LD_LIBRARY_PATH=`pwd`/build/FastFLIP:`pwd`/build/QuickOCT:`pwd`/build/ZMS:`pwd`/build/zenvdb:`pwd`/build/zenbase
=======
export PYTHONPATH=`pwd`
>>>>>>> master
if [ -z $USE_GDB ]; then
    python -m zenqt
else
    gdb python -ex 'r -m zenqt'
fi
