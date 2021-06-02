#!/bin/bash

killall python
ZSG_OPEN=assets/prim.zsg ZEN_SPROC=1 ZEN_DOEXEC=1 ./run.sh &

sleep 1.5
echo ==============
pid="`pidof python`"
files="`cat /proc/$pid/maps | awk '{print $6;}' | grep '\.so' | grep -v 'lib/python3' | sort | uniq`"
kib="`du $files | awk '{print $1}' | paste -sd+ | bc`"
echo $files
echo $[$kib / 1024]MB

killall python
wait
