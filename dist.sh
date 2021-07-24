#!/bin/bash
set -e

docker build . -t archibate/zeno_build -f Dockerfile.archlinux
docker run -v `pwd`:/tmp/zeno -v /tmp/release:/tmp/release -it archibate/zeno_build /root/runme.sh
