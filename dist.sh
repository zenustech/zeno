#!/bin/bash

rm -rf /tmp/dist
mkdir -p /tmp/dist/lib
mkdir -p /tmp/dist/sbin
mkdir -p /tmp/dist/bin
cp -rd build/*/*.so /tmp/dist/lib
cp -rd /opt/python3.9/bin/* /tmp/dist/sbin
cp -rd /opt/python3.9/lib/* /tmp/dist/lib
cp -rd /lib/libQt*.so* /tmp/dist/lib
cp -rd /lib/libcrypt.so* /tmp/dist/lib
cp -rd scripts/ldmock /tmp/dist/.ldmock
for x in /tmp/dist/sbin/*; do
    ln -s ../.ldmock /tmp/dist/bin/"$(basename "$x")"
done
