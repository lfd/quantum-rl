FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04

COPY . /app

RUN apt-get update && \
    apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install python3.8 python3-pip git -y

RUN ln -s /usr/bin/python3.8 /usr/bin/python

RUN python -m pip install --upgrade pip

WORKDIR /app

RUN pip3 install -r requirements.txt
