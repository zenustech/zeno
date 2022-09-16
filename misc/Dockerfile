FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update -y
RUN apt-get install -y build-essential cmake g++
RUN apt-get install -y git wget curl ninja vim
RUN apt-get install -y qt5-default libqt5svg5-dev
RUN apt-get install -y libblosc-dev libboost-iostreams-dev zlib1g-dev libtbb-dev
RUN apt-get install -y libeigen3-dev libcgal-dev liblapack-dev libopenblas-dev libhdf5-dev
RUN apt-get install -y libncurses5-dev libgdbm-dev libnss3-dev libssl-dev
RUN apt-get install -y libreadline-dev libffi-dev libsqlite3-dev libbz2-dev
#RUN curl https://github.com/archibate/vimrc/releases/download/v0.5/vimrc-install.sh | bash
WORKDIR /root
