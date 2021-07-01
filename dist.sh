#!/bin/bash
set -e

rm -rf /tmp/zenv
mkdir -p /tmp/zenv/lib

cp -d /usr/lib/ld-linux-x86-64.so.2 /usr/lib/ld-2.33.so /tmp/zenv/lib

cp -d `scripts/tracedll.sh python3 -m zenapi arts/lennardjones.zsg | awk -F.so '{print $1".so*";}'` /tmp/zenv/lib
