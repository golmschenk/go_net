# A Docker image for GoNet.
FROM python:3.5
MAINTAINER Greg Olmschenk <golmschenk@gmail.com>

ENV TENSORFLOW_VERSION 0.11.0rc0
ENV PYTHON_VERSION 35
ENV TF_GPU_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-${TENSORFLOW_VERSION}-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}m-linux_x86_64.whl
ENV TF_CPU_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-${TENSORFLOW_VERSION}-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}m-linux_x86_64.whl

RUN if which nvcc; then pip3 install --ignore-installed --upgrade ${TF_GPU_BINARY_URL}; else pip3 install --ignore-installed --upgrade ${TF_CPU_BINARY_URL}; fi
RUN pip3 install -e 'git+https://github.com/golmschenk/gonet#egg=gonet'