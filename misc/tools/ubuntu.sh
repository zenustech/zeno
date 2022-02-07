#!/bin/bash

set -e

rm -rf /tmp/build && mkdir /tmp/build && cd /tmp/build

apt-get install -y wget
apt-get install -y git
apt-get install -y gcc
apt-get install -y g++
apt-get install -y make
apt-get install -y cmake
apt-get install -y libboost-iostreams-dev
apt-get install -y libboost-system-dev
apt-get install -y libtbb-dev
apt-get install -y libilmbase-dev
apt-get install -y libopenexr-dev
apt-get install -y zlib1g-dev
apt-get install -y libeigen3-dev
apt-get install -y libopenblas-dev

git clone https://github.com/Blosc/c-blosc.git
cd c-blosc && git checkout tags/v1.5.0 -b v1.5.0 && mkdir build && cd build && cmake .. && make -j8 && make install && cd ../..

git clone https://github.com/AcademySoftwareFoundation/openvdb.git
cd openvdb && mkdir build && cd build && cmake .. && make -j8 && make install && cd ../..

wget http://www.netlib.org/blas/blast-forum/cblas.tgz
tar zxvf cblas.tgz && cd CBLAS && make -j8 && make install && cd ..
