import pandas as pd

csv_path = r"C:\Users\kikyo\ParlAI\data\blender90_preds.csv"
out_path = r"C:\Users\kikyo\ParlAI\data\mydata_eval_parlai.txt"

df = pd.read_csv(csv_path)

with open(out_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        label = str(row["label"]).strip()
        pred = str(row["prediction"]).strip()
        f.write(f"[text]: {text}\n")
        f.write(f"[labels]: {label}\n")
        f.write(f"[predicted]: {pred}\n")
        f.write("- - - - - - - END OF EPISODE - - - - - - - - - -\n")

print(f"✅ ParlAI 格式文件已生成：{out_path}")
