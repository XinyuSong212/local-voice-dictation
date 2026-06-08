# -*- coding: utf-8 -*-
"""快速自检: CUDA 是否可用, 不下载模型。"""
import voice_input  # 触发 _register_cuda_dlls()
import ctranslate2

n = ctranslate2.get_cuda_device_count()
print("ctranslate2 版本:", ctranslate2.__version__)
print("检测到 CUDA 设备数:", n)
print("=> 将使用:", "GPU (cuda/float16)" if n > 0 else "CPU (int8)")
