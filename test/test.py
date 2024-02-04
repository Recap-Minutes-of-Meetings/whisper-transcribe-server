import asyncio
import random
import websockets
import numpy as np
import librosa
import time

async def send_audio_chunks(uri, file_path):
    async with websockets.connect(uri) as websocket:
        data, _ = librosa.load(file_path, sr=16000)
        bin_data = data.tobytes()
        total_time = 0
        for i in range(random.randint(5, 15)):
            start_time = time.time()
            cur_num = random.randint(1, 3)
            for j in range(cur_num):
                # print(f'sent {file_path} {i} epoch, {j} iter')
                await websocket.send(bin_data)
            for j in range(cur_num):
                res = await websocket.recv()
                # print(f'recv {file_path}: {res} {i} epoch, {j} iter')
            end_time = time.time()
            epoch_time = end_time - start_time
            total_time += epoch_time
            # print(f"Epoch {i}: {epoch_time} seconds")
        
        avg_time = total_time / (i + 1)
        print(f"Average time for {file_path}: {avg_time} seconds")


async def main():
    uri = "ws://localhost:9090"
    file_paths = ["test/1.wav", "test/2.wav", "test/3.wav"]
    
    tasks = []
    for i in range(5):
        file_index = i % len(file_paths)
        file_path = file_paths[file_index]
        task = asyncio.create_task(send_audio_chunks(uri, file_path))
        tasks.append(task)

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
