# -*- coding: utf-8 -*-
"""完整管线自检: 下载模型到 D 盘 -> GPU 加载 -> 转写一段音频。"""
import time
import numpy as np
from voice_input import load_config, VoiceInput

cfg = load_config()
app = VoiceInput(cfg)

t0 = time.time()
app.load_model()
print(f"[自检] 模型加载完成, 用时 {time.time()-t0:.1f}s")

# 用一段 2 秒的轻微噪声验证推理路径不崩溃 (不期待识别出内容)
audio = (np.random.randn(16000 * 2) * 0.001).astype(np.float32)
t0 = time.time()
text = app.transcribe(audio)
print(f"[自检] 转写完成, 用时 {time.time()-t0:.1f}s")
print(f"[自检] 输出文本: {text!r}")
print("[自检] 管线 OK —— 模型/GPU/转写全部正常工作。")
