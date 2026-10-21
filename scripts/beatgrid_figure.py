"""Draw docs/beatgrid.png: two seconds of the synthetic drum track, where the kicks and hats
really are, and where the naive full-band grid and cutonbeat's grid put the beat.

    python3 scripts/beatgrid_figure.py docs/beatgrid.png
"""
import os, sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cutonbeat.core import SR, beat_grid
from cutonbeat.synth import drum_track

BPM, SECONDS, T0, SPAN = 124.0, 90.0, 45.0, 2.0  # window sits in the loud part, where the hats play
W, H, PAD = 1600, 560, 40
BG, INK, MUTED = (255, 255, 255), (30, 30, 30), (150, 150, 150)
KICK, HAT, NAIVE, GOOD = (30, 30, 30), (150, 150, 150), (214, 69, 65), (34, 139, 84)

y = drum_track(BPM, SECONDS)
P = 60 / BPM
naive, _, _ = beat_grid(y, kick_phase=False)
good, _, _ = beat_grid(y)


def ms_off(beats):
    return np.median(np.abs(((beats + P / 2) % P) - P / 2)) * 1000


def x_of(t):
    return PAD + (t - T0) / SPAN * (W - 2 * PAD)


def in_window(ts):
    return [t for t in ts if T0 <= t <= T0 + SPAN]


img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)
font = ImageFont.load_default(size=26)
small = ImageFont.load_default(size=22)

# Waveform envelope, one column per pixel.
top, mid, amp = 150, 250, 90
seg = y[int(T0 * SR):int((T0 + SPAN) * SR)]
cols = np.array_split(np.abs(seg), W - 2 * PAD)
for i, c in enumerate(cols):
    a = float(c.max()) * amp
    d.line([(PAD + i, mid - a), (PAD + i, mid + a)], fill=(200, 200, 200))

# Ground truth from how the track was built: kick on every beat, hat on every off-beat.
k0 = np.ceil(T0 / P)
kicks = in_window((k0 + np.arange(8)) * P)
hats = in_window((k0 - 0.5 + np.arange(9)) * P)
for t in kicks:
    d.line([(x_of(t), top), (x_of(t), mid + amp + 10)], fill=KICK, width=3)
for t in hats:
    d.line([(x_of(t), top + 40), (x_of(t), mid + amp - 30)], fill=HAT, width=2)


def grid_row(beats, yy, colour, label):
    for t in in_window(beats):
        d.polygon([(x_of(t), yy), (x_of(t) - 11, yy + 22), (x_of(t) + 11, yy + 22)], fill=colour)
    d.text((PAD, yy + 30), label, fill=colour, font=small)


grid_row(naive, 380, NAIVE, f"naive full-band grid: median {ms_off(naive):.0f} ms from the kick (on the hi-hat)")
grid_row(good, 450, GOOD, f"cutonbeat (kick-band phase): median {ms_off(good):.0f} ms from the kick")

d.text((PAD, 30), f"Where should the cut land?  Synthetic drum track, {BPM:.0f} BPM, "
       f"{T0:.0f}-{T0 + SPAN:.0f} s", fill=INK, font=font)
d.text((PAD, 75), "black = kick (the beat)    grey = off-beat hi-hat    triangles = where each grid cuts",
       fill=MUTED, font=small)

out = sys.argv[1] if len(sys.argv) > 1 else "beatgrid.png"
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
img.save(out, optimize=True)
print(f"{out}: naive {ms_off(naive):.0f} ms, cutonbeat {ms_off(good):.0f} ms")
