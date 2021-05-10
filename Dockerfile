FROM archlinux

COPY scripts/mirrorlist /etc/pacman.d/
RUN pacman -Sy
RUN pacman --noconfirm -S cmake python-pip
RUN pacman --noconfirm -S git vim
RUN pacman --noconfirm -S make gcc
RUN pacman --noconfirm -S glew glfw

COPY python/requirements.txt /root/
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install numpy pybind11

RUN echo git clone https://gitee.com/archibate/zeno.git --branch=nodep --depth=1 > /root/get-zeno.sh && chmod +x /root/get-zeno.sh

ENTRYPOINT bash
