FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3

COPY . /app

RUN python -m pip install --upgrade pip

WORKDIR /app

RUN pip3 install -r requirements.txt