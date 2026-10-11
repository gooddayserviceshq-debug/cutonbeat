"""Render the demo from synthetic media: no downloads, no copyrighted audio or footage.

    python3 scripts/demo.py demo-out
"""
import os, subprocess, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cutonbeat.synth import drum_track, write_wav, test_clip

out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "demo-out")
src = os.path.join(out, "src")
os.makedirs(os.path.join(src, "clips"), exist_ok=True)
song = write_wav(os.path.join(src, "song.wav"), drum_track(124, 75))
for i, pat in enumerate(["testsrc2", "smptehdbars", "rgbtestsrc", "testsrc", "mandelbrot", "life"]):
    test_clip(os.path.join(src, "clips", f"{i:02d}_{pat}.mp4"), seconds=8, pattern=pat, hue=i * 50)
sys.exit(subprocess.call([sys.executable, "-m", "cutonbeat", "--song", song, "--clips", os.path.join(src, "clips"),
                          "--out", os.path.join(out, "render"), "--title", "Test Pattern", "--artist", "cutonbeat",
                          "--end-text", "github.com/gooddayserviceshq-debug/cutonbeat", "--short-seconds", "15"],
                         cwd=os.path.join(os.path.dirname(__file__), "..")))
