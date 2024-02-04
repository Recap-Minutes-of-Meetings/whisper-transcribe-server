# Whisper Transcriber Server
Whisper websockets server with batched inference using FlashAttention-2 within Docker

## Requirements
- CUDA 12.1-compatible GPU.
- 8 GB of RAM.
- Installed `docker` with `nvidia-container-toolkit` installed.

## How to run
```shell
docker build -t recap/transcriber:latest .
docker run --name transcriber --gpus all -v ./data:/app/data recap/transcriber:latest
```

## How to use
- Connect to `ws://localhost:9090` via websocket and send binary audio chunks
in format of `np.ndarray` with `dtype=np.float32`, and `sampling_rate=16000`.
- You will receive `{"id": "The sequence number of the sent chunk", "text": "Transcribed text"}`.

Check out the `test/test.py` file for more clear usage example.

## Experiments results
- The model used: large-v3
- GPU used: RTX 4060 Laptop
- Batch size: 8
- VRAM constant usage: 3.1 GB
- VRAM usage during tests: 6 GB

- Flash attention 2 decreases average time of inference by 25-35%
- Batching decreases average time of inference for randomly sent chunks by 35-45%