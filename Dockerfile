FROM archlinux

COPY scripts/mirrorlist /etc/pacman.d/
COPY python/requirements.txt /root/
RUN pacman -Sy && pacman --noconfirm -S cmake python-pip
#RUN pacman -Sy && tbb cblas glm glew glfw boost openvdb eigen openblas lapack
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && pip install -r /root/requirements.txt
RUN git clone https://gitee.com/archibate/zeno.git --branch=nodep

ENTRYPOINT bash
