import asyncio
import json
import threading
import websockets
import torch
import os
import numpy as np

from transformers import pipeline
from datasets import Dataset, Audio
from transformers.pipelines.pt_utils import KeyDataset

from queue import Queue
from dataclasses import dataclass

WS_PORT = 9090
BATCH_SIZE = 8
SAMPLE_RATE = 16000
CACHE_DIR = "data"

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.float16,
    device_map="cuda:0",
    model_kwargs={"attn_implementation": "flash_attention_2"},
)


@dataclass
class AudioChunk:
    data: np.ndarray
    id: int  # the id of chunk for the websocket connection
    websocket: any


chunk_queue: Queue[AudioChunk] = Queue()


async def connection_handler(websocket, path):
    current_id = 0
    async for data in websocket:
        chunk_queue.put(AudioChunk(np.frombuffer(data, dtype=np.float32), current_id, websocket))
        current_id += 1


async def process_audio():
    while True:
        if chunk_queue.empty():
            await asyncio.sleep(0)
            continue

        chunks = []
        while not chunk_queue.empty() and len(chunks) < BATCH_SIZE:
            chunks.append(chunk_queue.get())

        data_chunks = [chunk.data for chunk in chunks]

        results = pipe (
            data_chunks,
            batch_size=BATCH_SIZE
        )

        # Sequential
        # results = [
        #     pipe(chunk)
        #     for chunk in data_chunks
        # ]

        # print(results)

        if len(results) != len(chunks):
            print('Length mismatch')

        for i, res in enumerate(results):
            chunk = chunks[i]
            asyncio.create_task(chunk.websocket.send(json.dumps({
                "id": chunk.id,
                "text": res["text"]
            })))


async def main():
    process_thread = threading.Thread(target=asyncio.run, args=(process_audio(),), daemon=True)
    process_thread.start()

    async with websockets.serve(connection_handler, "localhost", WS_PORT):
        await asyncio.Future()

    process_thread.join()


if __name__ == "__main__":
    asyncio.run(main())
