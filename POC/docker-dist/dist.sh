#!/bin/bash
set -e

sudo rm -rf /tmp/zenv
docker build distribute -t archibate/zeno_dist
docker run -v /tmp/zenv:/tmp/zenv -it archibate/zeno_dist /root/_dist.sh
