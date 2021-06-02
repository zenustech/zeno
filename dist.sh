#!/bin/bash

rm -rf /tmp/dist
mkdir -p /tmp/dist/lib
mkdir -p /tmp/dist/sbin
cp /lib/ld-linux-x86-64.so.2 /tmp/dist/lib

cp -rd /opt/python3.9/bin/* /tmp/dist/sbin
cp -rd /opt/python3.9/lib/* /tmp/dist/lib
cp -rd /lib/lib{crypt,c,m,pthread,util}{,-*}.so* /tmp/dist/lib

mkdir -p /tmp/dist/bin
cp -rd scripts/ldmock /tmp/dist/.ldmock
for x in /tmp/dist/sbin/*; do
    ln -s ../.ldmock /tmp/dist/bin/"$(basename "$x")"
done
