# -*- coding: utf-8 -*-
"""只下载必要的 SenseVoice 文件 (int8 模型 + tokens) 到 D 盘。"""
import os
from huggingface_hub import hf_hub_download

DEST = r"D:\ToolDev\voice-input\models\sensevoice"
REPO = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
os.makedirs(DEST, exist_ok=True)

for fn in ("model.int8.onnx", "tokens.txt"):
    print(f"下载 {fn} ...")
    p = hf_hub_download(repo_id=REPO, filename=fn, local_dir=DEST)
    print(f"  -> {p}  ({os.path.getsize(p)/1024/1024:.1f} MB)")

print("SenseVoice 模型就绪。")
