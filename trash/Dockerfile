# archibate/zeno
#
# ZENO application image
#
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build

RUN apt-get update
RUN apt-get upgrade -y

#################################
# Begin installing dependencies #
#################################

RUN apt-get install -y wget
RUN apt-get install -y git
RUN apt-get install -y gcc
RUN apt-get install -y g++
RUN apt-get install -y make
RUN apt-get install -y cmake
RUN apt-get install -y libboost-iostreams-dev
RUN apt-get install -y libboost-system-dev
RUN apt-get install -y libtbb-dev
RUN apt-get install -y libilmbase-dev
RUN apt-get install -y libopenexr-dev
RUN apt-get install -y zlib1g-dev
RUN apt-get install -y libeigen3-dev
RUN apt-get install -y libopenblas-dev

RUN git clone https://github.com/Blosc/c-blosc.git --depth=1 --branch=v1.5.0
RUN cd c-blosc && mkdir build && cd build && cmake .. && make -j32 && make install && cd ../..

RUN git clone https://github.com/AcademySoftwareFoundation/openvdb.git --depth=1 --branch=v7.2.1
RUN cd openvdb && mkdir build && cd build && cmake .. && make -j32 && make install && cd ../..

#################################################
# Below is only for end-user application images #
#################################################

RUN apt-get install -y python-is-python3
RUN apt-get install -y python-dev-is-python3
RUN apt-get install -y python3-pip

RUN apt-get install -y libglvnd-dev
RUN apt-get install -y libglapi-mesa

RUN rm -rf /build

WORKDIR /root
ENTRYPOINT bash
