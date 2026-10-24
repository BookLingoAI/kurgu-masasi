"""Kurulum sonrası doğrulama: her bileşeni gerçekten içe aktarır / çalıştırır. Çıkış kodu 0 = hepsi tamam."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ok = True


def check(name, fn, required=True):
    global ok
    try:
        print(f"[OK] {name}: {fn()}", flush=True)
    except Exception as e:  # noqa: BLE001
        if required:
            ok = False
        print(f"[{'X' if required else '!'}] {name}: {type(e).__name__}: {e}", flush=True)


def ffmpeg():
    import subprocess
    from kurgu.integrations.ffmpeg import ffmpeg_bin
    exe = ffmpeg_bin()
    out = subprocess.run([exe, "-hide_banner", "-version"], capture_output=True, text=True, creationflags=0x08000000)
    return out.stdout.splitlines()[0]


def gpu():
    from kurgu.integrations.cuda_env import ensure_cuda_dlls
    ensure_cuda_dlls()
    import ctranslate2
    n = ctranslate2.get_cuda_device_count()
    return f"{n} NVIDIA GPU -> Whisper {'GPU' if n else 'CPU (daha yavaş)'}"


check("Python", lambda: sys.version.split()[0])
check("NumPy", lambda: __import__("numpy").__version__)
check("OpenCV", lambda: __import__("cv2").__version__)
check("MediaPipe", lambda: __import__("mediapipe").__version__)
check("SciPy", lambda: __import__("scipy").__version__)
check("faster-whisper", lambda: __import__("faster_whisper").__version__)
check("fastembed", lambda: __import__("fastembed").__version__, required=False)
check("Dosya seçme penceresi (tkinter)", lambda: __import__("tkinter").TkVersion, required=False)
check("FFmpeg", ffmpeg)
check("Ekran kartı", gpu, required=False)
check("Uygulama", lambda: (__import__("kurgu.web.server"), "yüklendi")[1])
print("SONUC:", "TAMAM" if ok else "EKSIK", flush=True)
sys.exit(0 if ok else 1)
