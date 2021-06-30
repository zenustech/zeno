# archibate/ubuntu:18.04
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

COPY dist.sh .

ENTRYPOINT bash
