"""cutonbeat: a song + a folder of clips -> a beat-cut music video, three Shorts, thumbnails,
and a report.json that checks the outputs it claims to have made.

    cutonbeat --song song.wav --clips clips/ --out renders/ [--title "..."] [--artist "..."]

Only numpy, Pillow and ffmpeg.
"""
import argparse, json, os, shutil, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import __version__
from .core import SR, beat_grid, energy_tiers, plan_cuts, loudest_windows

FPS = 24
W, H = 1920, 1080
TIERS = {"quiet": 0, "intro": 0, "outro": 0, "mid": 1, "verse": 1,
         "loud": 2, "prechorus": 2, "chorus": 2, "breakdown": 2, "drop": 2, "solo": 2}
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/impact.ttf",
]


def run(cmd):
    subprocess.run(cmd, check=True)


def need(tool):
    if not shutil.which(tool):
        sys.exit(f"cutonbeat needs {tool} on PATH")


def probe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def load_audio(path):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, "<i2").astype(np.float32) / 32768


def font(path, px):
    for p in [path] + FONT_CANDIDATES:
        if p and os.path.exists(p):
            return ImageFont.truetype(p, px)
    return ImageFont.load_default(size=px)


def load_pools(clips_dir):
    """Clips go in a loudness pool. clips.json (optional) maps file name -> quiet|mid|loud or a
    section name (verse, chorus, ...). Unmapped clips go in every pool."""
    mapping_file = os.path.join(clips_dir, "clips.json")
    mapping = json.load(open(mapping_file)) if os.path.exists(mapping_file) else {}
    files = sorted(f for f in os.listdir(clips_dir) if f.lower().endswith((".mp4", ".mov", ".webm", ".mkv")))
    if not files:
        sys.exit(f"no clips in {clips_dir}")
    pools = {0: [], 1: [], 2: []}
    for f in files:
        path = os.path.join(clips_dir, f)
        clip = {"path": path, "dur": probe_duration(path), "uses": 0}
        tag = str(mapping.get(f, "")).lower()
        for t in ([TIERS[tag]] if tag in TIERS else [0, 1, 2]):
            pools[t].append(clip)
    for t in pools:  # a tier with no mapped clips borrows from the neighbours
        if not pools[t]:
            pools[t] = pools[1] or pools[2] or pools[0]
    return pools, len(files)


def text_png(path, size, lines, font_path, scrim=0):
    """Text rendered with Pillow, so ffmpeg builds without drawtext still work. scrim is the
    peak alpha of a soft dark band behind the text so it reads over busy footage."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    if scrim:
        yy = np.linspace(-1, 1, size[1])[:, None]
        band = np.zeros((size[1], size[0], 4), np.uint8)
        band[..., 3] = np.broadcast_to((scrim * np.exp(-(yy / 0.45) ** 2)).astype(np.uint8), (size[1], size[0]))
        img = Image.fromarray(band, "RGBA")
    d = ImageDraw.Draw(img)
    y = (size[1] - sum(px for _, px in lines) - 20 * (len(lines) - 1)) // 2
    for text, px in lines:
        f = font(font_path, px)
        d.text(((size[0] - d.textlength(text, font=f)) / 2, y), text, font=f, fill=(245, 242, 235, 255),
               stroke_width=max(2, px // 18), stroke_fill=(0, 0, 0, 255))
        y += px + 20
    img.save(path)
    return path


def streams(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height",
                          "-show_entries", "format=duration", "-of", "json", p], capture_output=True, text=True).stdout
    j = json.loads(out or "{}")
    v = next((s for s in j.get("streams", []) if s["codec_type"] == "video"), {})
    return {"duration": float(j.get("format", {}).get("duration", 0)),
            "streams": [s["codec_type"] for s in j.get("streams", [])],
            "size": [v.get("width"), v.get("height")]}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cutonbeat", description=__doc__.split("\n")[0])
    ap.add_argument("--song", required=True, help="audio file (wav/mp3/m4a)")
    ap.add_argument("--clips", required=True, help="folder of video clips; optional clips.json for sections")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="", help="title card + thumbnail text")
    ap.add_argument("--artist", default="")
    ap.add_argument("--end-text", default="", help="end card line, e.g. a channel handle")
    ap.add_argument("--watermark", default="", help="PNG placed bottom-right")
    ap.add_argument("--font", default="", help="TTF for cards and thumbnails")
    ap.add_argument("--shorts", type=int, default=3)
    ap.add_argument("--short-seconds", type=float, default=30)
    ap.add_argument("--version", action="version", version=__version__)
    a = ap.parse_args(argv)
    for tool in ("ffmpeg", "ffprobe"):
        need(tool)
    os.makedirs(a.out, exist_ok=True)
    work = os.path.join(a.out, "_work")
    os.makedirs(work, exist_ok=True)
    artist, title = a.artist.upper(), a.title.upper()

    mono = load_audio(a.song)
    total = len(mono) / SR
    beats, bpm, kicks = beat_grid(mono)
    tiers, energy = energy_tiers(mono, beats)
    cuts = plan_cuts(beats, tiers, total, kicks, 60 / bpm)
    pools, n_clips = load_pools(a.clips)
    print(f"{total:.1f}s, {bpm:.0f} bpm, {len(beats)} beats, {len(cuts)} cuts, {n_clips} clips")

    # Frame-exact segments: boundaries come from cumulative time so audio never drifts.
    seg_list, rot = [], {0: 0, 1: 0, 2: 0}
    for k, c in enumerate(cuts):
        nf = round(c["end"] * FPS) - round(c["start"] * FPS)
        if nf <= 0:
            continue
        pool = pools[c["tier"]]
        clip = pool[rot[c["tier"]] % len(pool)]
        rot[c["tier"]] += 1
        span = max(0.0, clip["dur"] - nf / FPS - 0.05)
        off = (clip["uses"] * 1.7) % span if span > 0 else 0.0  # a reused clip shows a different moment
        clip["uses"] += 1
        seg = os.path.join(work, f"seg_{k:04d}.mp4")
        run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{off:.3f}", "-i", clip["path"], "-an",
             "-vf", f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},setsar=1,"
                    f"tpad=stop_mode=clone:stop_duration=2",
             "-frames:v", str(nf), "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p", seg])
        c.update({"clip": os.path.basename(clip["path"]), "offset": round(off, 3), "frames": nf})
        seg_list.append(seg)
    concat = os.path.join(work, "concat.txt")
    with open(concat, "w") as fh:
        fh.writelines(f"file '{os.path.abspath(s)}'\n" for s in seg_list)
    video_only = os.path.join(work, "video_only.mp4")
    run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", concat, "-c", "copy", video_only])

    # Master: optional title card (0.6-5s), end card (last 6s) and watermark.
    inputs, chain, last, n_in = ["-i", video_only, "-i", a.song], [], "0", 2
    end_t = max(0.0, total - 6)
    if title or artist:
        png = text_png(os.path.join(work, "title.png"), (W, H), [(t, px) for t, px in ((artist, 150), (title, 84)) if t],
                       a.font, scrim=170)
        inputs += ["-loop", "1", "-i", png]
        chain += [f"[{n_in}]format=rgba,fade=in:st=0.6:d=0.8:alpha=1,fade=out:st=4.2:d=0.8:alpha=1[t]",
                  f"[{last}][t]overlay=enable='between(t,0.6,5)'[v{n_in}]"]
        last, n_in = f"v{n_in}", n_in + 1
    if a.end_text or artist:
        png = text_png(os.path.join(work, "end.png"), (W, H), [(t, px) for t, px in ((artist, 120), (a.end_text, 56)) if t],
                       a.font, scrim=170)
        inputs += ["-loop", "1", "-i", png]
        chain += [f"[{n_in}]format=rgba,fade=in:st={end_t:.2f}:d=0.8:alpha=1[e]",
                  f"[{last}][e]overlay=enable='gte(t,{end_t:.2f})'[v{n_in}]"]
        last, n_in = f"v{n_in}", n_in + 1
    mark = None
    if a.watermark:
        mark = os.path.join(work, "mark.png")
        im = Image.open(a.watermark).convert("RGBA")
        im.thumbnail((170, 170))
        im.save(mark)
        inputs += ["-i", mark]
        chain += [f"[{n_in}]format=rgba,colorchannelmixer=aa=0.7[m]", f"[{last}][m]overlay=W-w-40:H-h-36[v{n_in}]"]
        last, n_in = f"v{n_in}", n_in + 1
    chain += [f"[{last}]fade=in:st=0:d=0.5,fade=out:st={total - 1.2:.2f}:d=1.2[v]",
              f"[1:a]afade=out:st={total - 1.5:.2f}:d=1.5[a]"]
    master = os.path.join(a.out, "master_16x9.mp4")
    run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-filter_complex", ";".join(chain),
         "-map", "[v]", "-map", "[a]", "-t", f"{total:.3f}", "-r", str(FPS),
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-c:a", "aac", "-b:a", "256k", master])

    # Shorts: loudest non-overlapping windows, bar-aligned, never inside the title card.
    win = int(round(a.short_seconds / (60 / bpm)))
    shorts = []
    band = text_png(os.path.join(work, "short_band.png"), (1080, 300),
                    [(t, px) for t, px in ((artist, 104), (title, 56)) if t] or [("", 10)], a.font)
    for n, i in enumerate(loudest_windows(energy, beats, win, a.shorts), 1):
        st = float(beats[i])
        out = os.path.join(a.out, f"short_{n}_9x16.mp4")
        fc = ("[0]split[a][b];[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
              "boxblur=30:2,eq=brightness=-0.12[bg];[b]crop=ih*3/4:ih,scale=1080:1440[fg];"
              "[bg][fg]overlay=0:300[v1];[v1][2]overlay=0:0[v2]")
        extra = ["-i", band]
        if mark:
            extra += ["-i", mark]
            fc += ";[3]format=rgba,colorchannelmixer=aa=0.75[m];[v2][m]overlay=W-w-36:H-h-60[v]"
        else:
            fc += ";[v2]null[v]"
        fc += f";[1:a]afade=in:d=0.3,afade=out:st={a.short_seconds - 1:.2f}:d=1[au]"
        run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{st:.3f}", "-t", f"{a.short_seconds:.3f}", "-i", video_only,
             "-ss", f"{st:.3f}", "-t", f"{a.short_seconds:.3f}", "-i", a.song, *extra,
             "-filter_complex", fc, "-map", "[v]", "-map", "[au]", "-r", str(FPS),
             # Input -t alone is not enough: on the concat-copied intermediate the video stream
             # can run up to a segment past the cut. Bound the output too.
             "-t", f"{a.short_seconds:.3f}",
             "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
             "-movflags", "+faststart", out])
        shorts.append({"file": os.path.basename(out), "start": round(st, 2), "seconds": a.short_seconds})

    # Thumbnails + contact sheet from cut midpoints. Score = sharpness x contrast, loud cuts favoured.
    frames = []
    for k, c in enumerate(cuts):
        t = (c["start"] + c["end"]) / 2
        if "clip" not in c or t < 5.5 or t > end_t:
            continue
        p = os.path.join(work, f"f_{k:04d}.jpg")
        run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", video_only, "-frames:v", "1", "-q:v", "2", p])
        g = np.asarray(Image.open(p).convert("L").resize((480, 270)), dtype=np.float32)
        lap = np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]).var()
        frames.append((lap * (1 + 0.5 * c["tier"]) * g.std(), p, c["clip"]))
    cols = 6
    if frames:
        thumbs = [Image.open(p).resize((320, 180)) for _, p, _ in frames]
        sheet = Image.new("RGB", (320 * cols, 180 * ((len(thumbs) + cols - 1) // cols)))
        for i, im in enumerate(thumbs):
            sheet.paste(im, ((i % cols) * 320, (i // cols) * 180))
        sheet.save(os.path.join(a.out, "contact_sheet.jpg"), quality=85)
    seen, picks = set(), []
    for _, p, clip in sorted(frames, reverse=True):
        if clip not in seen:
            seen.add(clip)
            picks.append(p)
        if len(picks) == 3:
            break
    for n, p in enumerate(picks, 1):
        im = Image.open(p).convert("RGB").resize((1280, 720))
        if title or artist:
            g = np.clip(np.linspace(-0.2, 1, 720), 0, 1)[:, None] * np.clip(np.linspace(1.3, 0, 1280), 0, 1)[None, :]
            im = Image.composite(Image.new("RGB", im.size), im, Image.fromarray((g * 200).astype(np.uint8), "L"))
            d = ImageDraw.Draw(im)
            if title:
                d.text((48, 470), title, font=font(a.font, 132), fill=(255, 255, 255), stroke_width=8, stroke_fill=(0, 0, 0))
            if artist:
                d.text((54, 400), artist, font=font(a.font, 64), fill=(255, 122, 40), stroke_width=5, stroke_fill=(0, 0, 0))
        im.save(os.path.join(a.out, f"thumb_{n}.jpg"), quality=92)

    # QC from the artifacts, not exit codes: probe what was written and say what's wrong.
    m = streams(master)
    report = {"version": __version__, "song_seconds": round(total, 2), "bpm": round(bpm, 1),
              "cuts": len(seg_list), "clips_used": n_clips, "master": m,
              "shorts": [dict(s, **{"probe": streams(os.path.join(a.out, s["file"]))}) for s in shorts],
              "thumbs": [f"thumb_{n}.jpg" for n in range(1, len(picks) + 1)]}
    problems = []
    if abs(m["duration"] - total) > 0.25:
        problems.append(f"master duration {m['duration']:.2f} != song {total:.2f}")
    if sorted(m["streams"]) != ["audio", "video"]:
        problems.append(f"master streams {m['streams']}")
    if m["size"] != [W, H]:
        problems.append(f"master size {m['size']}")
    if len(shorts) < a.shorts:
        problems.append(f"only {len(shorts)} of {a.shorts} shorts (song too short for {a.short_seconds:.0f}s windows?)")
    for s in report["shorts"]:
        if s["probe"]["size"] != [1080, 1920] or abs(s["probe"]["duration"] - a.short_seconds) > 0.5:
            problems.append(f"{s['file']}: {s['probe']['size']} {s['probe']['duration']:.2f}s")
    report["problems"] = problems
    json.dump(cuts, open(os.path.join(a.out, "cuts.json"), "w"), indent=1)
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=2)
    print(json.dumps({k: report[k] for k in ("bpm", "cuts", "problems")}))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
