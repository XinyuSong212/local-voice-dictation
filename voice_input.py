"""
全局热键语音听写工具 (本地离线 Whisper)
按住热键说话 -> 松开后转写 -> 文字自动输入到当前光标处。

依赖: faster-whisper sounddevice numpy keyboard pyperclip
GPU:  nvidia-cublas-cu12 nvidia-cudnn-cu12
"""

import json
import os
import queue
import sys
import threading
import time

# 让控制台正确显示中文 (Windows 默认 cp936 会乱码)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 让 ctranslate2 在 Windows 上能找到 pip 安装的 CUDA / cuDNN DLL。
# 必须在 import ctranslate2 / faster_whisper 之前执行。
# ---------------------------------------------------------------------------
def _register_cuda_dlls():
    if not hasattr(os, "add_dll_directory"):
        return
    try:
        import nvidia
    except ImportError:
        return
    # nvidia 是 PEP420 命名空间包, 没有 __file__, 要用 __path__
    bases = list(getattr(nvidia, "__path__", []))
    for base in bases:
        for sub in ("cublas", "cudnn"):
            bin_dir = os.path.join(base, sub, "bin")
            if os.path.isdir(bin_dir):
                try:
                    os.add_dll_directory(bin_dir)
                except OSError:
                    pass
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


_register_cuda_dlls()

import keyboard
import numpy as np
import pyperclip
import sounddevice as sd

try:
    import winsound

    def beep(freq, dur):
        try:
            winsound.Beep(freq, dur)
        except RuntimeError:
            pass
except ImportError:  # 非 Windows

    def beep(freq, dur):
        pass


SAMPLE_RATE = 16000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

DEFAULTS = {
    "hotkey": "f9",
    "mode": "toggle",
    "engine": "whisper",  # "whisper" | "sensevoice"
    "beam_size": 1,
    "model": "large-v3",
    "model_dir": os.path.join(BASE_DIR, "models"),
    "device": "auto",
    "compute_type": "auto",
    "language": None,
    "initial_prompt": "以下是普通话和英文混合的句子。",
    # SenseVoice (可选引擎) 相关
    "sensevoice_dir": os.path.join(BASE_DIR, "models", "sensevoice"),
    "sensevoice_language": "auto",
    "use_itn": True,
    "num_threads": 4,
    "suppress_hotkey": True,
    "paste_mode": True,
    "restore_clipboard": True,
    "beep": True,
    "min_record_seconds": 0.3,
}


def load_config():
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception as e:
            print(f"[警告] 读取 config.json 失败,使用默认配置: {e}")
    # 把所有 HuggingFace 缓存也固定到 D 盘,避免任何文件落到 C 盘
    model_dir = cfg.get("model_dir")
    if model_dir:
        hf_home = os.path.join(model_dir, "hf")
        os.environ["HF_HOME"] = hf_home
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")
    return cfg


# ---------------------------------------------------------------------------
# 录音器: 按住热键期间持续采集麦克风音频
# ---------------------------------------------------------------------------
class Recorder:
    def __init__(self):
        self._frames = []
        self._stream = None
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):
        if status:
            # 溢出等非致命警告
            pass
        with self._lock:
            self._frames.append(indata.copy())

    def start(self):
        with self._lock:
            self._frames = []
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            if not self._frames:
                return np.zeros(0, dtype=np.float32)
            audio = np.concatenate(self._frames, axis=0).flatten()
        return audio


# ---------------------------------------------------------------------------
# 识别引擎 (可切换)
# ---------------------------------------------------------------------------
class SenseVoiceEngine:
    """SenseVoice (via sherpa-onnx): 体积小, 非自回归, 速度快, 中文/中英混合佳。"""

    name = "SenseVoice"

    def __init__(self, cfg):
        self.cfg = cfg
        self.recognizer = None

    def load(self):
        import sherpa_onnx

        d = self.cfg.get("sensevoice_dir")
        model = os.path.join(d, "model.int8.onnx")
        tokens = os.path.join(d, "tokens.txt")
        if not (os.path.exists(model) and os.path.exists(tokens)):
            raise FileNotFoundError(
                f"SenseVoice 模型缺失: {model}\n请先运行 python download_sensevoice.py"
            )
        print(f"[模型] 加载 SenseVoice (int8, CPU)  目录: {d}")
        t0 = time.time()
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model,
            tokens=tokens,
            num_threads=self.cfg.get("num_threads", 4),
            use_itn=self.cfg.get("use_itn", True),  # 逆文本归一化: 数字/标点
            language=self.cfg.get("sensevoice_language", "auto"),
        )
        print(f"[模型] 就绪,用时 {time.time() - t0:.1f}s")

    def transcribe(self, audio):
        s = self.recognizer.create_stream()
        s.accept_waveform(SAMPLE_RATE, audio)
        self.recognizer.decode_stream(s)
        return s.result.text.strip()


class WhisperEngine:
    """faster-whisper (large-v3): 多语种, GPU 加速, 体积大、相对慢。"""

    name = "Whisper"

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None

    @staticmethod
    def _detect_device():
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", "float16"
        except Exception:
            pass
        return "cpu", "int8"

    def load(self):
        from faster_whisper import WhisperModel

        device = self.cfg["device"]
        compute_type = self.cfg["compute_type"]
        if device == "auto":
            device, auto_compute = self._detect_device()
            if compute_type == "auto":
                compute_type = auto_compute
        elif compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        model_dir = self.cfg.get("model_dir")
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        print(f"[模型] 加载 {self.cfg['model']}  (device={device}, compute={compute_type})")
        t0 = time.time()
        try:
            self.model = WhisperModel(
                self.cfg["model"],
                device=device,
                compute_type=compute_type,
                download_root=model_dir,
            )
        except Exception as e:
            print(f"[模型] GPU 加载失败 ({e}),回退到 CPU ...")
            self.model = WhisperModel(
                self.cfg["model"],
                device="cpu",
                compute_type="int8",
                download_root=model_dir,
            )
            device = "cpu"
        print(f"[模型] 就绪,用时 {time.time() - t0:.1f}s  (实际 device={device})")

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(
            audio,
            language=self.cfg["language"],
            initial_prompt=self.cfg["initial_prompt"],
            beam_size=self.cfg.get("beam_size", 1),
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
            no_repeat_ngram_size=3,
        )
        return "".join(seg.text for seg in segments).strip()


def build_engine(cfg):
    engine = cfg.get("engine", "sensevoice").lower()
    if engine == "whisper":
        return WhisperEngine(cfg)
    return SenseVoiceEngine(cfg)


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
class VoiceInput:
    def __init__(self, cfg):
        self.cfg = cfg
        self.recorder = Recorder()
        self.is_recording = False
        self._key_held = False
        self.engine = build_engine(cfg)
        self.jobs = queue.Queue()
        self._stop = False

    # ---- 模型加载 -------------------------------------------------------
    def load_model(self):
        print(f"[引擎] 使用 {self.engine.name}")
        self.engine.load()

    # ---- 转写 -----------------------------------------------------------
    def transcribe(self, audio):
        if audio.size < SAMPLE_RATE * self.cfg["min_record_seconds"]:
            print("[转写] 录音太短,忽略。")
            return ""
        return self.engine.transcribe(audio)

    # ---- 输出文字 -------------------------------------------------------
    def output_text(self, text):
        if not text:
            return
        if self.cfg["paste_mode"]:
            saved = None
            if self.cfg["restore_clipboard"]:
                try:
                    saved = pyperclip.paste()
                except Exception:
                    saved = None
            pyperclip.copy(text)
            time.sleep(0.05)
            keyboard.send("ctrl+v")
            if saved is not None:
                time.sleep(0.2)
                try:
                    pyperclip.copy(saved)
                except Exception:
                    pass
        else:
            keyboard.write(text)

    # ---- 后台转写线程 ---------------------------------------------------
    def _worker(self):
        while not self._stop:
            try:
                audio = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                t0 = time.time()
                text = self.transcribe(audio)
                if text:
                    print(f"[结果] {text}    ({time.time() - t0:.1f}s)")
                    self.output_text(text)
                if self.cfg["beep"]:
                    beep(1200, 80)  # 完成提示音
            except Exception as e:
                print(f"[错误] 转写失败: {e}")
            finally:
                self.jobs.task_done()

    # ---- 录音开关 -------------------------------------------------------
    def _start_rec(self):
        self.is_recording = True
        self.recorder.start()
        if self.cfg["beep"]:
            beep(900, 60)  # 开始录音提示音
        print("[录音] 开始 ...")

    def _stop_rec(self):
        self.is_recording = False
        audio = self.recorder.stop()
        print(f"[录音] 结束 ({audio.size / SAMPLE_RATE:.1f}s),转写中...")
        if self.cfg["beep"]:
            beep(600, 60)  # 结束录音提示音
        self.jobs.put(audio)

    # ---- 热键回调: 切换模式 (按一下开始, 再按一下结束) -----------------
    def _on_toggle(self, event):
        if self._key_held:
            return  # 忽略系统自动重复, 一次物理按下只切换一次
        self._key_held = True
        if self.is_recording:
            self._stop_rec()
        else:
            self._start_rec()

    def _on_key_up(self, event):
        self._key_held = False

    # ---- 热键回调: 按住模式 (按住说话, 松开输入) -----------------------
    def _on_press(self, event):
        if self.is_recording:
            return
        self._start_rec()

    def _on_release(self, event):
        if not self.is_recording:
            return
        self._stop_rec()

    # ---- 运行 -----------------------------------------------------------
    def run(self):
        self.load_model()

        worker = threading.Thread(target=self._worker, daemon=True)
        worker.start()

        hotkey = self.cfg["hotkey"]
        suppress = self.cfg["suppress_hotkey"]
        mode = self.cfg.get("mode", "toggle")

        if mode == "hold":
            keyboard.on_press_key(hotkey, self._on_press, suppress=suppress)
            keyboard.on_release_key(hotkey, self._on_release, suppress=suppress)
            how = f"按住 [{hotkey.upper()}] 说话,松开即输入。"
        else:
            keyboard.on_press_key(hotkey, self._on_toggle, suppress=suppress)
            keyboard.on_release_key(hotkey, self._on_key_up, suppress=suppress)
            how = f"按一下 [{hotkey.upper()}] 开始,再按一下 [{hotkey.upper()}] 结束并输入。"

        print("=" * 56)
        print(f"  语音听写已就绪。{how}")
        print("  按 Ctrl+C 退出。")
        print("=" * 56)
        if self.cfg["beep"]:
            beep(1500, 120)

        try:
            keyboard.wait()  # 阻塞至 KeyboardInterrupt (Ctrl+C)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop = True
            print("\n[退出] 再见。")


def main():
    cfg = load_config()
    app = VoiceInput(cfg)
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n[退出] 再见。")


if __name__ == "__main__":
    main()
