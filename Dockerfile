FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /root

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list

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

RUN git clone https://gitee.com/codespace1212/c-blosc.git
RUN cd c-blosc && git checkout tags/v1.5.0 -b v1.5.0 && mkdir build && cd build && cmake .. && make -j8 && make install && cd ../..

RUN git clone https://gitee.com/zeng_gui/openvdb.git
RUN cd openvdb && mkdir build && cd build && cmake .. && make -j8 && make install && cd ../..

#################################################
# Below is only for end-user application images #
#################################################

RUN apt-get install -y python-is-python3
RUN apt-get install -y python-dev-is-python3
RUN apt-get install -y python3-pip
RUN python -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

RUN python -m pip install numpy
RUN python -m pip install PyQt5
RUN apt-get install -y libqt5core5a
RUN apt-get install -y qt5dxcb-plugin
RUN apt-get install -y libglvnd-dev
RUN apt-get install -y libglapi-mesa
RUN apt-get install -y libosmesa6

ENTRYPOINT bash
