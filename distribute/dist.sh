#!/bin/bash
set -e

rm -rf /tmp/zenv
docker build . -t archibate/zeno_dist
docker run -v /tmp/zenv:/tmp/zenv -it archibate/zeno_dist /root/_dist.sh
