#!/bin/bash

export PYTHONPATH=`pwd`/python
kill `lsof -i tcp:8000 | awk '{print $2}' | grep -v PID | uniq` 2> /dev/null
python -m zenserver &
python -m zenclient
kill %1
wait
