FROM nvcr.io/nvidia/pytorch:23.04-py3

ENV DEBIAN_FRONTEND=noninteractive

ENV MPLBACKEND=agg

RUN apt-get update && \
        apt-get install -y \
        git \
        wget \
        unzip \
        vim \
        zip \
        curl \
        yasm \
        pkg-config \
        nano \
        tzdata \
        ffmpeg \
        libgtk2.0-dev \
        libgl1-mesa-glx && \
    rm -rf /var/cache/apk/*

RUN pip install --upgrade pip

RUN pip --no-cache-dir install \
      Cython==0.29.21

RUN pip --no-cache-dir install \
    numpy==1.23.1 \
	matplotlib==3.7.1 \
	tqdm==4.65.0 \
    pillow==9.2.0 \
	opencv-python==4.5.5.64 \
	tensorboard==2.9.0 \
	pyyaml \
    pytorch-lightning==1.9.2 \
    setuptools==65.5.1 \
    scikit-image==0.21.0 \
    tensorboardX \
    optuna==3.6.1 \
    timm==1.0.3

RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

RUN apt-get update
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y
RUN apt-get update
RUN apt-get upgrade libstdc++6 -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
       cd torch2trt && python3 setup.py install
RUN pip install hydra-core==1.3.2
RUN ln -sf /usr/share/zoneinfo/Turkey /etc/localtime

