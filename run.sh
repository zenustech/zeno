#!/bin/bash

export PYTHONPATH=`pwd`/python
python -m zenedit &
build/zenvis/zenvis
wait
