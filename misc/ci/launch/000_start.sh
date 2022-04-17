#!/bin/bash

chmod +x "`dirname $0`"/usr/bin/*
exec "`dirname $0`"/usr/bin/zenoedit "$@"
