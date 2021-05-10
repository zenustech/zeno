FROM archlinux

COPY scripts/mirrorlist /etc/pacman.d/
COPY python/requirements.txt /root/
RUN pacman -Sy
RUN pacman --noconfirm -S cmake python-pip
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r /root/requirements.txt
RUN pacman --noconfirm -S git vim
RUN pacman --noconfirm -S make gcc
RUN pacman --noconfirm -S glew glfw
CMD git clone https://gitee.com/archibate/zeno.git --branch=nodep --depth=1

ENTRYPOINT bash
