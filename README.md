# Local Voice Dictation

Press a hotkey, speak, press again — the recognized text is typed into **whatever
app holds the cursor** (chat, Word, browser, anything). Runs fully offline and
handles mixed Chinese + English.

## Features

- **Global hotkey** — tap `F9` to start, tap `F9` again to stop and type (toggle mode; stable)
- **Fully offline** — default engine is `faster-whisper` `large-v3`, GPU-accelerated on NVIDIA; nothing is uploaded
- **Mixed Chinese/English** — recognizes Mandarin with embedded English
- **Two engines** — Whisper (accurate, GPU) or SenseVoice (smaller and faster; see below)
- **Auto type** — pastes via the clipboard so non-ASCII text is never dropped, then restores your previous clipboard
- **Portable model storage** — models live in `./models` next to the tool

## Installation

```powershell
python -m pip install -r requirements.txt
```

GPU acceleration needs an NVIDIA card; the CUDA libraries (`nvidia-cublas-cu12`,
`nvidia-cudnn-cu12`) are listed in `requirements.txt`. Without a GPU the tool
falls back to CPU automatically.

## Usage

Double-click **`启动语音听写.bat`**, or run:

```powershell
python voice_input.py
```

The first launch downloads the model (Whisper `large-v3`, ~3 GB) into `./models`.
Once you see the "ready" banner:

1. Click into any text field
2. **Tap F9** to start recording (a mid beep plays), then speak
3. **Tap F9** again to stop — within a second the text appears at the cursor

Press `Ctrl+C` to quit.

> Prefer push-to-talk (hold to speak, release to type)? Set `mode` to `"hold"` in `config.json`.

## Engines

Switch with the `engine` field in `config.json`:

| Engine | Notes | Model size | Speed |
|--------|-------|-----------|-------|
| `whisper` (default) | faster-whisper `large-v3`, GPU, multilingual | ~3 GB | ~1 s/utterance |
| `sensevoice` | SenseVoice (sherpa-onnx), CPU, strong on Chinese/mixed | ~230 MB | faster |

To use SenseVoice, first download its model, then set `"engine": "sensevoice"` and restart:

```powershell
python download_sensevoice.py
```

## Configuration

Copy `config.example.json` to `config.json` and edit. All keys are optional;
omitted keys fall back to built-in defaults.

| Key | Meaning | Default |
|-----|---------|---------|
| `hotkey` | Trigger key | `"f9"` |
| `mode` | `"toggle"` (tap/tap) or `"hold"` (push-to-talk) | `"toggle"` |
| `engine` | `"whisper"` or `"sensevoice"` | `"whisper"` |
| `model` | Whisper size: `tiny/base/small/medium/large-v3` | `"large-v3"` |
| `device` | `auto` / `cuda` / `cpu` | `"auto"` |
| `compute_type` | `auto` / `float16` / `int8` | `"auto"` |
| `language` | `null` = auto-detect; force Chinese with `"zh"` | `null` |
| `model_dir` | Where models are stored | `./models` |
| `suppress_hotkey` | Block the hotkey's normal action in the active app | `true` |
| `paste_mode` | `true` = clipboard paste (robust for CJK); `false` = type char by char | `true` |
| `beep` | Play feedback beeps | `true` |

Common tweaks:

- **Too slow / low VRAM** — set `model` to `medium` or `small`
- **Hotkey conflicts** — change `hotkey` to `f8`, `right ctrl`, etc.
- **English misheard as the wrong language** — set `language` to `"zh"`

## Troubleshooting

- **F9 does nothing** — the `keyboard` library needs admin rights for a global hook. Right-click `启动语音听写.bat` → Run as administrator.
- **CUDA / cuDNN errors** — the tool falls back to CPU (slower); confirm `nvidia-cudnn-cu12` and `nvidia-cublas-cu12` are installed.
- **Paste does nothing or is garbled** — some apps block Ctrl+V; set `paste_mode` to `false`.
- **No audio captured** — check the Windows default microphone.

## Self-check

```powershell
python test_gpu.py        # check GPU availability (no model download)
python test_pipeline.py   # download model + load + transcribe test audio
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
