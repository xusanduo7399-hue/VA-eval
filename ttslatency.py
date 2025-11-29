
import time
import pandas as pd
import pyttsx3
from multiprocessing import Process, Queue, freeze_support


def synthesize_in_process(text, out_q):
    """在子进程中运行 pyttsx3，避免 Windows SAPI 死锁"""
    engine = pyttsx3.init()
    engine.save_to_file(text, "tmp_pytts.wav")
    engine.runAndWait()
    out_q.put(True)


def measure_latency(text):
    """测量一条文本的 latency"""
    q = Queue()
    p = Process(target=synthesize_in_process, args=(text, q))

    t0 = time.time()
    p.start()
    q.get()     # 等待子进程结束
    p.join()
    t1 = time.time()

    return t1 - t0


if __name__ == "__main__":
    freeze_support()  # Windows 必须加这一行

    # ------------------------------
    # 1. 加载 dataset
    # ------------------------------
    df = pd.read_csv(
        r"C:\Users\kikyo\ParlAI\data\blender90_preds.csv",
        encoding="latin1"
    )

    TEXT_FIELD = "prediction"  # 你要测的字段
    texts = df[TEXT_FIELD].astype(str).tolist()

    results = []

    # ------------------------------
    # 2. 批量测量 latency
    # ------------------------------
    for idx, text in enumerate(texts):
        latency = measure_latency(text)

        results.append({
            "id": idx,
            "text": text,
            "text_len": len(text),
            "tts_latency_sec": latency
        })

        if idx < 5:
            print(f"[{idx}] len={len(text)} latency={latency:.4f} text={text[:60]}...")

    # ------------------------------
    # 3. 保存结果
    # ------------------------------
    out_df = pd.DataFrame(results)
    out_df.to_csv("pyttsx3_latency_results.csv", index=False, encoding="utf-8-sig")

    print("\nSaved to pyttsx3_latency_results.csv")



