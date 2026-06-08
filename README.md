# 语音听写工具 (本地离线)

按一下热键说话,再按一下自动把识别出的文字输入到**当前光标所在的任意软件**(微信、Word、浏览器、聊天框……)。完全本地离线运行,支持中英文混合。

## 特点

- **全局热键**:默认按 `F9` 开始,再按 `F9` 结束并输入(切换模式,稳定)
- **本地离线**:默认 `faster-whisper` 的 `large-v3`,在 NVIDIA GPU 上加速,不联网、不上传
- **中英混合**:自动识别普通话 + 英文夹杂
- **双引擎可选**:Whisper(准、GPU)或 SenseVoice(更快更小,见下文)
- **自动输入**:通过剪贴板粘贴,中文不丢字(会自动恢复你原来的剪贴板内容)
- **模型存放在 D 盘**,不占用 C 盘

## 安装

```powershell
cd D:\ToolDev\voice-input
python -m pip install -r requirements.txt
```

## 使用

双击 **`启动语音听写.bat`**,或在终端运行:

```powershell
python voice_input.py
```

首次启动会自动下载模型(Whisper large-v3 约 3GB,下到 D 盘)。看到 `语音听写已就绪` 后:

1. 把光标点到任意输入框
2. **按一下 F9** 开始录音(会有"嘀"提示音),正常说话
3. **再按一下 F9** 结束,等不到一秒,文字就自动出现在光标处

按 `Ctrl+C` 退出。

> 想要"按住说话、松开输入"的体验?把 `config.json` 的 `mode` 改成 `"hold"`。

## 识别引擎

`config.json` 的 `engine` 字段切换:

| 引擎 | 说明 | 模型大小 | 速度 |
|------|------|---------|------|
| `whisper` (默认) | faster-whisper large-v3,GPU 加速,多语种 | ~3 GB | 约 1s/句 |
| `sensevoice` | SenseVoice(sherpa-onnx),CPU,中文/中英混合佳 | ~230 MB | 更快 |

切到 SenseVoice 前先下载它的模型:

```powershell
python download_sensevoice.py
```

然后把 `config.json` 里 `"engine"` 改成 `"sensevoice"` 重启即可。

## 提示音含义

| 声音 | 含义 |
|------|------|
| 中音"嘀" | 开始录音 |
| 高音"嘀" | 转写完成、文字已输入 |
| 三连音(启动时) | 程序就绪 |

## 配置 (`config.json`)

| 字段 | 说明 | 默认 |
|------|------|------|
| `hotkey` | 说话热键 | `"f9"` |
| `model` | 模型大小,可选 `tiny/base/small/medium/large-v3` | `"large-v3"` |
| `device` | `auto` / `cuda` / `cpu` | `"auto"` |
| `compute_type` | `auto` / `float16` / `int8` | `"auto"` |
| `language` | `null`=自动检测;锁定中文填 `"zh"` | `null` |
| `initial_prompt` | 偏置提示,帮助中英混合识别 | 中英混合提示 |
| `model_dir` | 模型存放目录(D 盘) | `D:\ToolDev\voice-input\models` |
| `suppress_hotkey` | 是否屏蔽热键在当前软件的原本功能 | `true` |
| `paste_mode` | `true`=剪贴板粘贴(推荐,中文稳);`false`=逐字输入 | `true` |
| `beep` | 是否播放提示音 | `true` |

改完配置重启程序生效。

### 常见调整

- **嫌慢/显存不够**:把 `model` 改成 `medium` 或 `small`
- **热键和别的软件冲突**:把 `hotkey` 改成 `f8`、`right ctrl` 等
- **识别老把中文听成英文**:把 `language` 设成 `"zh"`

## 故障排查

- **按 F9 没反应**:`keyboard` 库需要管理员权限才能全局监听。右键 `启动语音听写.bat` → 以管理员身份运行。
- **报 CUDA / cudnn 相关错误**:程序会自动回退到 CPU(较慢)。确认 `nvidia-cudnn-cu12`、`nvidia-cublas-cu12` 已安装。
- **粘贴出来是乱码或没反应**:个别软件禁用了 Ctrl+V,可把 `paste_mode` 设为 `false` 试逐字输入。
- **没声音输入设备 / 录不到音**:检查 Windows 默认麦克风设置。

## 自检

```powershell
python test_gpu.py        # 检查 GPU 是否可用 (不下模型)
python test_pipeline.py   # 下载模型 + 加载 + 转写测试音频
```
