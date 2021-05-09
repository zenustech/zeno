#!/bin/bash

export PYTHONPATH=`pwd`/python
export LD_LIBRARY_PATH=`pwd`/build/FastFLIP:`pwd`/build/QuickOCT:`pwd`/build/MPM
python -m zenclient
