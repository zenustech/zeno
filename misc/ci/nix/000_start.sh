#!/bin/bash

set -e
echo "-- Welcome to report bug at: https://github.com/zenustech/zeno/issues"
chmod +x "`dirname $0`"/usr/bin/*
exec "`dirname $0`"/usr/bin/zenoedit "$@"
