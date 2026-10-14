# Adapted from AI Engineering from Scratch (MIT), phases/06-speech-and-audio/02-spectrograms-mel-features
"""Waveform -> frames -> Hann window -> STFT -> mel filterbank -> log-mel -> MFCC, in NumPy.

The audio is SYNTHETIC: a 0.5 s chirp sweeping 200 Hz to 4,000 Hz plus a 2 kHz tone burst,
sampled at 16 kHz. No audio files or downloads.

Run from guide/:  uv run code/29-speech-and-audio/mel_features.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import plt, save  # noqa: E402

SR = 16_000           # samples per second
WIN, HOP = 400, 160   # 25 ms window, 10 ms hop at 16 kHz
N_FFT = 512           # zero-pad each frame to 512 for the FFT -> 257 frequency bins
N_MELS = 80


def synth(sr=SR, seconds=1.0):
    t = np.arange(int(sr * seconds)) / sr
    half = t < 0.5
    f0, f1, T = 200.0, 4000.0, 0.5
    chirp = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * T) * t ** 2))   # instantaneous freq f0 + (f1-f0) t/T
    tone = np.sin(2 * np.pi * 2000.0 * t)
    return 0.5 * np.where(half, chirp, tone) * (t < 0.9)                  # last 0.1 s silent


# ---------- step 1: frame ----------
def frame(x, win=WIN, hop=HOP):
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]           # [n_frames, win]
    return x[idx]


# ---------- step 2: window + FFT ----------
def hann(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)              # periodic Hann (as torch/librosa)


def power_spectrum(frames, n_fft=N_FFT):
    spec = np.fft.rfft(frames * hann(frames.shape[1]), n=n_fft)           # [n_frames, n_fft//2 + 1]
    return np.abs(spec) ** 2


# ---------- step 3: mel filterbank ----------
def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sr=SR, fmin=0.0, fmax=None):
    fmax = fmax or sr / 2
    edges_hz = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))  # equally spaced in mel
    bin_hz = np.arange(n_fft // 2 + 1) * sr / n_fft                                    # centre of each FFT bin
    fb = np.zeros((n_mels, len(bin_hz)))
    for m in range(n_mels):
        lo, c, hi = edges_hz[m], edges_hz[m + 1], edges_hz[m + 2]
        rise = (bin_hz - lo) / (c - lo)
        fall = (hi - bin_hz) / (hi - c)
        fb[m] = np.maximum(0, np.minimum(rise, fall))                     # triangle peaking at 1 at c
    return fb, edges_hz


# ---------- step 4: log-mel and MFCC ----------
def log_mel(power, fb, eps=1e-10):
    return np.log10(np.maximum(power @ fb.T, eps))                      # [n_frames, n_mels]


def dct_ii(x, n_coeffs=13):
    N = x.shape[-1]
    n = np.arange(N)
    basis = np.cos(np.pi * np.arange(n_coeffs)[:, None] * (2 * n + 1) / (2 * N))   # [n_coeffs, N]
    return x @ basis.T


def main():
    x = synth()
    frames = frame(x)
    print(f"signal: {len(x)} samples at {SR} Hz ({len(x) / SR:.1f} s)")
    print(f"frames: 1 + ({len(x)} - {WIN}) // {HOP} = {frames.shape[0]}  -> shape {frames.shape}")

    w = hann(WIN)
    print(f"Hann: w[0] = {w[0]:.3f}, w[{WIN // 2}] = {w[WIN // 2]:.3f}, w[{WIN - 1}] = {w[-1]:.5f}")

    P = power_spectrum(frames)
    print(f"power spectrum shape {P.shape}; bin spacing {SR / N_FFT:.2f} Hz")
    f_tone = 70                                                           # a frame in the 2 kHz tone part
    k = int(P[f_tone].argmax())
    print(f"frame {f_tone} (t = {f_tone * HOP / SR:.2f} s): peak bin {k} -> {k * SR / N_FFT:.1f} Hz (true 2000 Hz)")

    print("\nmel scale: m = 2595 log10(1 + f/700)")
    for f in [0, 700, 1000, 4000, 8000]:
        print(f"  {f:>5} Hz -> {float(hz_to_mel(f)):7.1f} mel")

    fb, edges = mel_filterbank()
    widths = edges[2:] - edges[:-2]
    print(f"\nfilterbank shape {fb.shape}; filter 0 spans {edges[0]:.0f}-{edges[2]:.0f} Hz ({widths[0]:.0f} Hz wide), "
          f"filter 79 spans {edges[79]:.0f}-{edges[81]:.0f} Hz ({widths[79]:.0f} Hz wide)")

    L = log_mel(P, fb)
    M = dct_ii(L)
    print(f"log-mel shape {L.shape}; MFCC shape {M.shape}")
    print("chirp check: frame -> strongest mel filter -> its centre frequency vs the true chirp frequency")
    for i in [5, 15, 25, 35, 45]:
        m = int(L[i].argmax())
        t_mid = (i * HOP + WIN / 2) / SR
        print(f"  frame {i:2d} (t = {t_mid:.3f} s): mel {m:2d}, centre {edges[m + 1]:6.0f} Hz, "
              f"true {200 + 3800 * t_mid / 0.5:6.0f} Hz")
    print(f"silent frame 95: max log-mel {L[95].max():.1f} (floor log10(1e-10) = -10)")

    # sanity check: our DCT matches the direct formula for one frame
    direct = [sum(L[30, n] * np.cos(np.pi * kk * (2 * n + 1) / (2 * N_MELS)) for n in range(N_MELS)) for kk in range(3)]
    assert np.allclose(direct, M[30, :3])

    # ---------- figure ----------
    t_frames = (np.arange(frames.shape[0]) * HOP + WIN / 2) / SR
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    ax = axes[0, 0]
    seg = 2 * HOP + WIN                                                    # first three frames
    tt = np.arange(seg) / SR * 1000
    ax.plot(tt, x[:seg], lw=0.8, color="#a1a1aa", label="waveform (chirp start)")
    for i, c in enumerate(["#2563eb", "#0d9488", "#7c3aed"]):
        s0 = i * HOP
        ax.plot((s0 + np.arange(WIN)) / SR * 1000, 0.5 * hann(WIN), color=c, lw=1.5, label=f"Hann window, frame {i}")
    ax.set_title(f"First 3 frames: 25 ms windows every 10 ms (of {len(x):,} samples)")
    ax.set_xlabel("time (ms)")
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[0, 1]
    ax.imshow(10 * np.log10(P.T + 1e-10), origin="lower", aspect="auto", cmap="magma",
              extent=[t_frames[0], t_frames[-1], 0, SR / 2], vmin=-60)
    ax.set_title(f"Log power spectrogram: {P.shape[0]} frames x {P.shape[1]} bins")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.grid(False)
    ax = axes[1, 0]
    for m in range(0, N_MELS, 4):
        ax.plot(np.arange(fb.shape[1]) * SR / N_FFT, fb[m], lw=0.8)
    ax.set_title("Every 4th mel filter: narrow at low Hz, wide at high Hz")
    ax.set_xlabel("frequency (Hz)")
    ax = axes[1, 1]
    ax.imshow(L.T, origin="lower", aspect="auto", cmap="magma", extent=[t_frames[0], t_frames[-1], 0, N_MELS], vmin=-6)
    ax.set_title(f"Log-mel spectrogram: {L.shape[0]} frames x {N_MELS} mels")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel filter index")
    ax.grid(False)
    fig.tight_layout()
    save(fig, "spectrograms-mel-features")


if __name__ == "__main__":
    main()
