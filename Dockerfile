FROM nvcr.io/nvidia/cuda:12.1.0-devel-ubi8 as base
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install flash-attn --no-build-isolation
RUN pip install accelerate

COPY ./src .

FROM base as runtime
EXPOSE 9090
CMD ["python3", "src/server.py"]
