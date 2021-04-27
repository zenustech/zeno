#!/bin/bash

export PYTHONPATH=`pwd`/python
kill `lsof -i tcp:8000 | awk '{print $2}' | grep -v PID | uniq` 2> /dev/null
xvfb-run python -m zenserver
