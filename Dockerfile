FROM archlinux

COPY scripts/mirrorlist /etc/pacman.d/
COPY python/requirements.txt /root/
RUN pacman -Sy && pacman -S cmake python-pip pybind11 tbb cblas glm glew glfw boost openvdb eigen openblas lapack --noconfirm
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && pip install -r /root/requirements.txt

ENTRYPOINT bash
