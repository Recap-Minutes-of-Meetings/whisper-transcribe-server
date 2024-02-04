FROM nvcr.io/nvidia/cuda:12.0.1-devel-ubuntu22.04 as base
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt update && \
    apt install -y \
        git \
        python3-pip \
        python3-dev

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install flash-attn --no-build-isolation
RUN pip install accelerate

COPY ./src ./src

FROM base as runtime
EXPOSE 9090
CMD ["python3", "src/server.py"]
