# archibate/zeno_dev
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /root

RUN apt-get update
RUN apt-get upgrade -y
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
RUN apt-get install -y dh-autoreconf
RUN apt-get install -y libffi-dev

RUN git clone https://github.com/zensim-dev/c-blosc.git
RUN cd c-blosc && git checkout tags/v1.5.0 -b v1.5.0 && mkdir build && cd build && cmake .. && make -j32 && make install && cd ../..

RUN git clone https://github.com/zensim-dev/openvdb.git
RUN cd openvdb && mkdir build && cd build && cmake .. && make -j32 && make install && cd ../..

RUN apt-get install -y python-is-python3
RUN apt-get install -y python-dev-is-python3
RUN apt-get install -y python3-pip

ENTRYPOINT bash
