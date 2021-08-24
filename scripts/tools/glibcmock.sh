#!/bin/bash

echo "#if defined(__linux__)"
strings /lib/libc.so.6 | grep @GLIBC | awk -F@ '{print "__asm__(\".symver "$1","$0"\");"}' | grep -v '@@' | grep GLIBC_2.2.5
echo "#endif"
