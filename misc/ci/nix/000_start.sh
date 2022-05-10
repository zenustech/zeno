#!/bin/bash

set -e
echo "-- Welcome to report bug at: https://github.com/zenustech/zeno/issues"
echo "-- Trouble shooting & FAQs: https://github.com/zenustech/zeno/blob/master/docs/FAQ.md"
chmod +x "$(dirname "$0")"/usr/bin/*
echo "-- You can view the log file at: /tmp/zeno-$$.log"
if [ "x$DISPLAY" == "x" ]; then echo "==> ERROR: you don't have X11 DISPLAY! WSL user please install XLaunch first (see docs/FAQ.md)"; fi
"$(dirname "$0")"/usr/bin/zenoedit "$@" 2>&1 | tee /tmp/zeno-$$.log
