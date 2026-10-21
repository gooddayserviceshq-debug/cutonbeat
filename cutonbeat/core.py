"""Audio analysis and cut planning. Pure numpy: no ffmpeg, no files, so it is testable.

The pipeline:
  onset_env    spectral flux, locally detrended, so a sustained pad doesn't read as a hit
  beat_grid    coarse tempo by autocorrelation, then a fine period+phase search so the grid
               doesn't drift over a 3-minute song; also returns kick onsets (30-150 Hz flux peaks)
  energy_tiers per-beat RMS, smoothed over two bars, split into quiet / mid / loud
  plan_cuts    cut on beats, snapped to a real kick when one is near; quiet sections hold a shot
               for 8 beats, loud ones cut every 2
"""
import numpy as np

SR = 22050
HOP, NFFT = 512, 2048
# Flux frame i compares windows i and i+1; an onset registers when it reaches the centre of
# window i+1. Without this shift every beat and kick lands ~70 ms early.
LATENCY = (NFFT / 2 + HOP) / SR
BEATS_PER_CUT = {0: 8, 1: 4, 2: 2}


def onset_env(mono, sr=SR, lo=None, hi=None):
    """Positive spectral flux per hop, minus its local mean. Returns (envelope, frames per second)."""
    if len(mono) < NFFT * 2:
        raise ValueError(f"audio too short: {len(mono) / sr:.2f}s")
    frames = np.lib.stride_tricks.sliding_window_view(mono, NFFT)[::HOP] * np.hanning(NFFT)
    S = np.abs(np.fft.rfft(frames, axis=1))
    if lo is not None:
        f = np.fft.rfftfreq(NFFT, 1 / sr)
        S = S[:, (f >= lo) & (f < hi)]
    S = np.log1p(S)
    flux = np.maximum(0, np.diff(S, axis=0)).sum(1)
    return np.maximum(flux - np.convolve(flux, np.ones(16) / 16, "same"), 0), sr / HOP


def beat_grid(mono, sr=SR, bpm_range=(70, 190), kick_phase=True):
    """Returns (beat times in seconds, bpm, kick onset times in seconds).

    Tempo comes from full-band flux. Phase comes mostly from the kick band: full-band flux
    weights broadband hi-hats as heavily as kicks, so on a track with off-beat hats it locks
    the grid onto the "and" of every beat, half a beat away from where a cut should land.
    kick_phase=False gives that naive full-band grid, for comparison only."""
    flux, fps = onset_env(mono, sr)
    kick, _ = onset_env(mono, sr, 30, 150)
    # Kick-band weighting only when there is a kick to weight; ambient tracks fall back to flux.
    has_kick = kick_phase and kick.sum() > 0.02 * flux.sum()
    phase_env = flux / (flux.max() + 1e-9) + (3 * kick / (kick.max() + 1e-9) if has_kick else 0)
    ac = np.correlate(flux, flux, "full")[len(flux) - 1:]
    lags = np.arange(len(ac)) / fps
    ok = (lags > 60 / bpm_range[1]) & (lags < 60 / bpm_range[0])
    step0 = lags[ok][np.argmax(ac[ok])] * fps
    x = np.arange(len(flux))
    best = (-1.0, step0, 0.0)
    for step in np.linspace(step0 * 0.97, step0 * 1.03, 121):
        for p in np.linspace(0, step, 24, endpoint=False):
            score = np.interp(np.arange(p, len(flux) - 1, step), x, phase_env).sum()
            if score > best[0]:
                best = (score, step, p)
    _, step, phase = best
    thr = np.percentile(kick, 90)
    peaks = (kick > thr) & (kick >= np.roll(kick, 1)) & (kick >= np.roll(kick, -1))
    kicks = np.where(peaks)[0] / fps + LATENCY
    return np.arange(phase, len(flux), step) / fps + LATENCY, 60 * fps / step, kicks


def snap(t, kicks, tol):
    """Move a cut onto the nearest real kick if one is within tol seconds."""
    if not len(kicks):
        return float(t)
    k = kicks[np.argmin(np.abs(kicks - t))]
    return float(k) if abs(k - t) <= tol else float(t)


def energy_tiers(mono, beats, sr=SR):
    """Per-beat loudness tier (0 quiet, 1 mid, 2 loud) and the smoothed energy curve."""
    rms = np.array([np.sqrt(np.mean(mono[int(a * sr):int(b * sr)] ** 2) + 1e-12)
                    for a, b in zip(beats[:-1], beats[1:])])
    smooth = np.convolve(rms, np.ones(8) / 8, "same")
    lo, hi = np.percentile(smooth, [30, 70])
    return np.where(smooth >= hi, 2, np.where(smooth >= lo, 1, 0)), smooth


def plan_cuts(beats, tiers, total, kicks, beat_len, beats_per_cut=BEATS_PER_CUT):
    """List of {start, end, tier}. Covers [0, total] exactly, no gaps, no overlaps."""
    cuts, i = [], 0
    while i < len(beats) - 1:
        tier = int(tiers[min(i, len(tiers) - 1)])
        cuts.append({"start": snap(beats[i], kicks, beat_len / 4), "tier": tier})
        i = min(i + beats_per_cut[tier], len(beats) - 1)
    for a, b in zip(cuts, cuts[1:]):
        a["end"] = b["start"]
    cuts[0]["start"] = 0.0
    cuts[-1]["end"] = float(total)
    return [c for c in cuts if c["end"] > c["start"]]


def loudest_windows(energy, beats, win_beats, n=3, skip_seconds=6.0):
    """Start beat indices of the n loudest non-overlapping windows, bar-aligned (4 beats)."""
    if len(energy) <= win_beats:
        return []
    order = np.argsort([-energy[i:i + win_beats].mean() for i in range(len(energy) - win_beats)])
    taken = []
    for i in order:
        i = int(i) - int(i) % 4
        if beats[i] < skip_seconds or any(abs(i - j) < win_beats for j in taken):
            continue
        taken.append(i)
        if len(taken) == n:
            break
    return sorted(taken)
