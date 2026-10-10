"""Synthetic test media: a drum track with a known tempo and quiet/loud sections, and
colour-bar clips. Lets the tests and the demo run with no copyrighted audio or footage."""
import subprocess, wave
import numpy as np

from .core import SR


def drum_track(bpm=120.0, seconds=60.0, sr=SR, seed=0):
    """Kick on every beat, hat on off-beats, a pad. Loud (kick x3, hats on) from 40% to 80%."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    y = 0.05 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.25 * t))
    kt = np.arange(int(0.25 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 + 90 * np.exp(-kt * 30)) * kt) * np.exp(-kt * 14)
    hat = rng.standard_normal(int(0.04 * sr)) * np.exp(-np.arange(int(0.04 * sr)) / sr * 90)
    beat = 60.0 / bpm
    for k in range(int(seconds / beat)):
        s = int(k * beat * sr)
        loud = 0.4 * seconds <= k * beat < 0.8 * seconds
        seg = kick[: max(0, min(len(kick), n - s))]
        y[s:s + len(seg)] += (0.9 if loud else 0.3) * seg
        if loud:
            h = int((k + 0.5) * beat * sr)
            seg = hat[: max(0, min(len(hat), n - h))]
            y[h:h + len(seg)] += 0.15 * seg
    return (y / np.abs(y).max() * 0.9).astype(np.float32)


def write_wav(path, y, sr=SR):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def test_clip(path, seconds=6, size="640x360", pattern="testsrc2", hue=0):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"{pattern}=size={size}:rate=24",
                    "-t", str(seconds), "-vf", f"hue=h={hue}", "-pix_fmt", "yuv420p", path], check=True)
    return path
