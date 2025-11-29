import time
import pandas as pd
from parlai.core.agents import create_agent_from_model_file

# --------------------------------------------------------
# 路径配置
# --------------------------------------------------------
MODEL_PATH = r"C:\Users\kikyo\ParlAI\data\models\blender\blender_90M\model"
DATA_PATH = r"C:\Users\kikyo\ParlAI\data\mydata_parlai.txt"
OUTPUT_CSV = "blender_latency_from_parlai.csv"

print("Loading blender_90M model...")
agent = create_agent_from_model_file(MODEL_PATH)
print("Model loaded!\n")


# --------------------------------------------------------
# 加载 ParlAI 文本数据
# --------------------------------------------------------
def load_parlai_file(path):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("text:"):
                q = line[len("text:"):].strip()
                queries.append(q)
    return queries


queries = load_parlai_file(DATA_PATH)
print(f"Loaded {len(queries)} queries.\n")


# --------------------------------------------------------
# 测试 latency（关键修复：需要 episode_done=True）
# --------------------------------------------------------
latency_records = []

for i, query in enumerate(queries):

    # 必须：给 observe() 加 episode_done
    agent.observe({
        "text": query,
        "episode_done": True
    })

    t0 = time.time()
    reply = agent.act()
    t1 = time.time()

    latency = t1 - t0
    output_text = reply.get("text", "")

    latency_records.append({
        "id": i,
        "input": query,
        "output": output_text,
        "latency_sec": latency
    })

    if i < 5:
        print(f"Sample {i}")
        print("Input :", query)
        print("Output:", output_text)
        print("Latency:", round(latency, 4), "sec")
        print("-" * 50)


# --------------------------------------------------------
# 保存结果
# --------------------------------------------------------
df = pd.DataFrame(latency_records)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("\nSaved results to:", OUTPUT_CSV)
print("Average Latency:", df["latency_sec"].mean())
print("Max Latency:", df["latency_sec"].max())
print("Min Latency:", df["latency_sec"].min())
