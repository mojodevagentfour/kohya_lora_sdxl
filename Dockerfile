FROM tensorflow/tensorflow:2.7.4-gpu-jupyter

RUN python -m pip install --upgrade pip; \
    apt-get update; \
    pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 torchtext==0.15.1 torchdata==0.6.0 --extra-index-url https://download.pytorch.org/whl/cu118 -U; \
    pip install bitsandbytes; \
    apt-get install -y libunwind-dev \
    apt-get update; \
    apt --fix-broken install -y; \
    apt-get remove --purge aria2 -y; \
    apt-get autoremove -y; \
    apt-get install aria2 -y; \
    apt --fix-broken install;

# Set the working directory
WORKDIR /root

# Clone the necessary repositories
RUN git clone https://github.com/qaneel/kohya-trainer /root/content/kohya-trainer
RUN git clone https://github.com/zanllp/sd-webui-infinite-image-browsing.git /root/content/repositories/infinite-image-browsing
RUN git clone https://github.com/Linaqruf/discordia-archivum /root/content/repositories/discordia-archivum

# Set up the directory structure
RUN mkdir -p /root/content/drive/MyDrive && \
    mkdir /root/content/deps && \
    mkdir /root/content/LoRA && \
    mkdir /root/content/pretrained_model && \
    mkdir /root/content/vae && \
    mkdir /root/content/network_weight && \
    mkdir /root/content/LoRA/config

# Download and set up other dependencies
RUN apt-get install wget
RUN wget -c https://github.com/camenduru/gperftools/releases/download/v1.0/libtcmalloc_minimal.so.4 -O /root/content/libtcmalloc_minimal.so.4

# Install Python dependencies
WORKDIR /root/content/kohya-trainer/
RUN pip3 install --upgrade -r requirements.txt
RUN pip3 install python-dotenv

WORKDIR /root/content/repositories/infinite-image-browsing/
RUN pip3 install --upgrade -r requirements.txt

WORKDIR /root/content/repositories/discordia-archivum
RUN pip3 install --upgrade -r requirements.txt

WORKDIR /root

RUN pip install --upgrade pip
RUN apt-get update; \
    apt-get install -y \
    libcairo2 \
    libcairo2-dev; \
    apt-get update; \ 
    apt-get install -y libunwind8-dev libunwind8; \
    apt-get install google-auth-oauthlib==0.4.1; \
    apt --fix-broken install -y;

RUN pip install manimlib; \
    pip install manimce; \
    pip uninstall rich -y; \
    pip install boto3; \
    pip install xformers==0.0.19 triton==2.0.0 -U;

COPY kohya_lora_sdxl_trainer_v2.py /root/

RUN mkdir .aws
ADD .aws/* /root/.aws

RUN wget -c https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors -O /root/content/vae/sdxl_vae.safetensors

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# Clean up to reduce the image size
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*