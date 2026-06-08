# -*- coding: utf-8 -*-
"""冒烟测试: 不用麦克风, 验证两个引擎都能加载并跑通转写。"""
import numpy as np
from voice_input import load_config, build_engine

audio = (np.random.randn(16000 * 2) * 0.001).astype(np.float32)  # 2s 轻噪声

for eng_name in ("sensevoice", "whisper"):
    cfg = load_config()
    cfg["engine"] = eng_name
    print(f"\n==== 测试引擎: {eng_name} ====")
    eng = build_engine(cfg)
    eng.load()
    text = eng.transcribe(audio)
    print(f"[OK] {eng.name} 转写返回: {text!r}")

print("\n两个引擎均可用。")
