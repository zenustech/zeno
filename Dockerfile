FROM archlinux

COPY scripts/mirrorlist /etc/pacman.d/
RUN pacman --noconfirm -Sy
RUN pacman --noconfirm -S git
RUN pacman --noconfirm -S gcc
RUN pacman --noconfirm -S make
RUN pacman --noconfirm -S cmake
RUN pacman --noconfirm -S python
RUN pacman --noconfirm -S python-pip
RUN pacman --noconfirm -S pybind11
RUN pacman --noconfirm -S python-numpy
RUN pacman --noconfirm -S python-pyqt5
RUN pacman --noconfirm -S qt5-base
RUN pacman --noconfirm -S libglvnd
RUN pacman --noconfirm -S mesa
RUN pacman --noconfirm -S ttf-roboto

RUN python -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

RUN echo git clone https://gitee.com/archibate/zeno.git --depth=1 > /root/gitee-zeno.sh && chmod +x /root/gitee-zeno.sh
RUN echo git clone https://github.com/zensim-dev/zeno.git --depth=1 > /root/github-zeno.sh && chmod +x /root/github-zeno.sh

ENTRYPOINT bash
