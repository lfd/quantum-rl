FROM nvcr.io/nvidia/tensorflow:21.05-tf2-py3

COPY . /app
WORKDIR /app

RUN python -m pip install --upgrade pip

RUN pip3 install -r requirements.txt

# install pytorch
RUN pip3 install torch==1.10.1+cu113 \
    torchvision==0.11.2+cu113 \
    torchaudio==0.10.1+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html
