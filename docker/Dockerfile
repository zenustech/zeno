#FROM openvdb/vfx2019
FROM python:3.7

COPY sources.list /etc/apt/
RUN apt-get clean && apt-get update
#RUN apt-get install python-is-python3 python-dev-is-python3
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

ENTRYPOINT bash
