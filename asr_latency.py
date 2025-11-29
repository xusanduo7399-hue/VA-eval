import time
import pandas as pd
import whisper
from datasets import load_dataset
import numpy as np

print("Loading Whisper small...")
asr = whisper.load_model("small")
print("Whisper loaded.")

dataset = load_dataset("AudioLLMs/alpaca_audio_test", split="test")
dataset = dataset.with_format("python")

print("Dataset loaded:", len(dataset))

lat_results = []

for i in range(len(dataset)):
    sample = dataset[i]

    # numpy array + sampling rate
    audio_array = sample["context"]["array"]
    sampling_rate = sample["context"]["sampling_rate"]

    # ⭐ Whisper 只能处理 float32，而 dataset 是 float64
    audio_array = audio_array.astype(np.float32)

    t0 = time.time()
    result = asr.transcribe(audio_array, fp16=False)
    t1 = time.time()

    latency = t1 - t0

    lat_results.append({
        "id": i,
        "asr_latency": latency,
        "recognized_text": result["text"]
    })

    # print前几条检查
    if i < 5:
        print(f"\nSample {i}")
        print("ASR text:", result["text"])
        print("Latency:", round(latency, 4), "sec")

df = pd.DataFrame(lat_results)
df.to_csv("whisper_asr_latency.csv", index=False, encoding="utf-8-sig")

print("\nSaved whisper_asr_latency.csv")
print("Average ASR Latency:", df["asr_latency"].mean())
print("Max:", df["asr_latency"].max())
print("Min:", df["asr_latency"].min())
