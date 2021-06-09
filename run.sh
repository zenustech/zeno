#!/bin/bash

export PYTHONPATH=`pwd`
if [ -z $USE_GDB ]; then
    python -m zenqt
else
    gdb python -ex 'r -m zenqt'
fi
