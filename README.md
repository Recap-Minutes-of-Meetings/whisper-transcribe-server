# whisper-transcribe-server
Whisper websockets server with batched inference using flashattn2 within Docker

## Requirements
CUDA 12.1-compatible GPU.
8 GB of RAM.
Installed `docker` with `nvidia-container-toolkit` installed.

## How to run
```shell
docker build -t recap/transcriber:latest .
docker run --name transcriber --gpus all recap/transcriber:latest
```

## Experiments results
The model used: large-v3
GPU used: RTX 4060 Laptop
Batch size: 8
VRAM constant usage: 3.1 GB
VRAM usage during tests: 6 GB

Flash attention 2 decreases average time of inference by 25-35%
Batching decreases average time of inference for randomly sent chunks by 35-45%