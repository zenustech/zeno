# archibate/zeno_dist
FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /root

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y git
RUN apt-get install -y gcc
RUN apt-get install -y g++
RUN apt-get install -y make
RUN apt-get install -y cmake

RUN apt-get install -y python3-pip
RUN python3 -m pip install -U pip -i https://mirrors.aliyun.com/pypi/simple/
RUN python3 -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

RUN apt-get install -y libffi-dev
RUN apt-get install -y zlib1g-dev
RUN apt-get install -y patchelf

RUN apt-get install -y libboost-iostreams-dev
RUN apt-get install -y libboost-system-dev
RUN apt-get install -y libtbb-dev
RUN apt-get install -y libilmbase-dev
RUN apt-get install -y libopenexr-dev
RUN apt-get install -y libeigen3-dev
RUN apt-get install -y libopenblas-dev

RUN apt-get autoremove -y cmake
RUN apt-get install -y wget
RUN wget https://cmake.org/files/v3.12/cmake-3.12.2-Linux-x86_64.tar.gz
RUN tar zxvf cmake-3.12.2-Linux-x86_64.tar.gz
RUN mv cmake-3.12.2-Linux-x86_64 /opt/cmake-3.12.2
RUN ln -sf /opt/cmake-3.12.2/bin/* /usr/bin && cmake --version

RUN git clone https://gitee.com/codespace1212/c-blosc.git --depth=1 --branch=v1.5.0
RUN cd c-blosc && mkdir build && cd build && cmake .. && make -j32 && make install && cd ../..

RUN apt-get install -y libjemalloc-dev

RUN git clone https://gitee.com/zeng_gui/openexr.git --depth=1
RUN cd openexr && mkdir build && cd build && cmake .. && make -j32 && make install && cd ../..

RUN git clone https://gitee.com/zeng_gui/openvdb.git --depth=1
RUN cd openvdb && mkdir build && cd build && cmake .. && make -j32 && make install && cd ../..

COPY _dist.sh /root
