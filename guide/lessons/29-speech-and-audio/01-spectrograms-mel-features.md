---
title: Spectrograms and mel features
summary: Turn a waveform into the log-mel spectrogram that speech models consume (framing, windowing, FFT, mel filterbank, log, MFCC) and get the shapes and parameters right.
skill: deep-learning
minutes: 40
prerequisites: [vectors-and-matrices, convolutional-networks]
related: [speech-recognition-asr, recurrent-networks, transformer-architecture]
adapted_from: [phases/06-speech-and-audio/02-spectrograms-mel-features]
---

# Spectrograms and mel features

**Mental model.** A spectrogram is sheet music computed from sound. Cut the recording into short overlapping slices, ask "which pitches are loud in this slice?", and stack the answers left to right. A mel spectrogram then squeezes the pitch axis the way your ear does: fine detail at low pitches, coarse at high ones.

**You will learn to**

- Frame a waveform with a window and hop, and compute the number of frames and their shape.
- Explain why each frame is multiplied by a window (Hann) before the FFT.
- Convert between hertz and mel, build a triangular mel filterbank, and apply it as one matrix multiply.
- Produce log-mel features and MFCCs, and state the shapes at every stage.
- Pick sample rate, window, hop, and mel count for a speech model, and avoid the train/serve mismatches that break audio pipelines.

**Why it matters.** Nearly every speech recognizer, speech synthesizer, and audio classifier in 2026 either consumes log-mel spectrograms or was designed around them. Whisper's input is a log-mel spectrogram. When an audio model fails in production, the cause is often the front end: a different sample rate, log base, normalization, or padding from what the model saw in training. This lesson builds the front end from scratch so those knobs are no longer magic.

## 1. Intuition

**A waveform is the wrong shape for learning.** One second of audio at 16 kHz is 16,000 numbers between −1 and 1. The same vowel spoken twice gives completely different sample values (shifted by a millisecond, a bit louder), yet sounds the same. What stays stable is *which frequencies carry energy*, over windows of tens of milliseconds.

**From waveform to spectrogram.**

1. **Frame.** Cut the signal into short overlapping frames. The speech default is a 25 ms window every 10 ms: 400 samples with a hop of 160 at 16 kHz. Speech is roughly stable over 25 ms, and 10 ms steps track how fast it changes.
2. **Window.** Multiply each frame by a bell-shaped Hann window that fades to zero at both ends. Without it, the cut edges look like sudden jumps, and the FFT smears energy across many frequencies ("spectral leakage").
3. **FFT.** The Fourier transform of each frame says how much of each frequency it contains. Keep the magnitude (or its square, the power).
4. **Stack.** Frames become columns of a time × frequency image: the [spectrogram](../../glossary.md#spectrogram).

**The mel scale.** Hearing is roughly logarithmic in pitch: 100 Hz to 200 Hz sounds like a big step; 7,000 Hz to 7,100 Hz barely registers. The mel scale warps frequency to match. A **mel filterbank** is a set of overlapping triangular filters, equally spaced in mel, so they're narrow at low frequencies and wide at high ones. Each filter adds up the power in its range. 257 FFT bins become 80 mel bands: a [mel spectrogram](../../glossary.md#mel-spectrogram).

**Log.** Loudness perception is also roughly logarithmic, and raw power spans many orders of magnitude. Taking the log compresses that range so quiet consonants and loud vowels are both visible to the model.

**MFCC.** Applying a discrete cosine transform (DCT) to each log-mel frame and keeping the first 13 coefficients gives [mel-frequency cepstral coefficients](../../glossary.md#mfcc) (MFCCs). They were the standard feature for decades because they're compact and decorrelated, which suited Gaussian models. Neural networks do fine on log-mels directly, so modern ASR uses log-mels; MFCCs survive in small, classical, and some speaker-recognition systems.

> [!NOTE]
> **Not every model uses mels.** Self-supervised models such as wav2vec 2.0 and HuBERT learn their own front end from the raw waveform with a convolutional encoder. Neural audio codecs (used in many TTS and audio-LLM systems) also start from raw samples. But the log-mel spectrogram is still the most common input, and the concepts (frames, hops, frequency resolution) apply everywhere.

## 2. Visualization

![Four panels from a synthetic one-second signal. Top left: the first 45 ms of waveform with three overlapping Hann windows, 25 ms long and starting every 10 ms. Top right: log power spectrogram, 98 frames by 257 bins, showing a rising line from 200 to 4,000 Hz in the first half second, then a flat line at 2,000 Hz until 0.9 s, then silence; vertical streaks mark the abrupt switch and stop. Bottom left: every fourth of 80 triangular mel filters, narrow below 1 kHz and wide near 8 kHz. Bottom right: log-mel spectrogram, 98 frames by 80 mels, where the straight chirp in hertz becomes a curve that rises steeply then flattens.](../../figures/spectrograms-mel-features.png)

*Synthetic signal: a chirp sweeping 200 to 4,000 Hz for 0.5 s, a 2 kHz tone until 0.9 s, then silence, sampled at 16 kHz.*

Compare the two spectrograms. In hertz, the chirp is a straight line because its frequency rises linearly. In mels it bends: low frequencies get many mel bands, so the curve climbs fast at first, then high frequencies share fewer, wider bands, so it flattens. The vertical streaks at 0.5 s and 0.9 s are real: an abrupt change in a signal contains energy at all frequencies, which is also why clicks and plosives ("p", "t") look like vertical lines in speech spectrograms.

**Try it** (predict first, then edit `mel_features.py`):

1. Set `WIN = 1600` (100 ms). Predict what happens to the chirp line's thickness and to the sharpness of the 0.5 s transition.
2. Replace `hann(...)` with a window of all ones (no tapering). Predict the change in the tone's spectrogram row.
3. Set `N_MELS = 128`. Predict how many low filters cover just one FFT bin, and whether the chirp curve looks smoother.
4. Resample conceptually: change `SR` to 8,000 while keeping the chirp to 4,000 Hz. Predict what happens to the top of the sweep. (Hint: the highest frequency a sampled signal can represent is half the sample rate.)

## 3. The math

### Symbols

| Symbol | Meaning | Shape / unit |
|---|---|---|
| $x[n]$ | waveform sample $n$ | scalar, $n = 0 \dots L-1$ |
| $f_s$ | sample rate | Hz (samples per second) |
| $W$ | window length | samples |
| $H$ | hop length | samples |
| $F$ | number of frames | count |
| $w[n]$ | window function | length $W$ |
| $N_{\text{fft}}$ | FFT size (frame zero-padded to this) | samples |
| $P$ | power spectrogram | $F \times (N_{\text{fft}}/2 + 1)$ |
| $M$ | number of mel filters | count |
| $B$ | mel filterbank matrix | $M \times (N_{\text{fft}}/2 + 1)$ |
| $S$ | log-mel spectrogram | $F \times M$ |

### Framing

```math
F = 1 + \left\lfloor \frac{L - W}{H} \right\rfloor
```

Frame $t$ covers samples $tH$ to $tH + W - 1$.

### Window and FFT

Hann window (periodic form, as in PyTorch and librosa): $w[n] = 0.5 - 0.5\cos(2\pi n / W)$. Power spectrum of frame $t$ at bin $k$:

```math
P[t, k] = \left| \sum_{n=0}^{W-1} w[n]\, x[tH + n]\, e^{-2\pi i k n / N_{\text{fft}}} \right|^2, \quad k = 0, \dots, N_{\text{fft}}/2
```

Bin $k$ corresponds to frequency $k f_s / N_{\text{fft}}$. Only bins up to $N_{\text{fft}}/2$ are kept because a real signal's spectrum is symmetric.

### Mel scale and filterbank

```math
m(f) = 2595 \log_{10}\!\left(1 + \frac{f}{700}\right), \qquad f(m) = 700\left(10^{m/2595} - 1\right)
```

Place $M + 2$ edges equally spaced in mel from $m(f_{\min})$ to $m(f_{\max})$, convert back to hertz as $e_0, \dots, e_{M+1}$. Filter $j$ is a triangle rising from $e_j$ to a peak of 1 at $e_{j+1}$ and falling to 0 at $e_{j+2}$. For a bin at frequency $f$:

```math
B[j, k] = \max\!\left(0, \min\!\left(\frac{f - e_j}{e_{j+1} - e_j}, \frac{e_{j+2} - f}{e_{j+2} - e_{j+1}}\right)\right)
```

Then the whole mel step is one matrix multiply and a log:

```math
S = \log_{10}\!\big(\max(P B^\top, \epsilon)\big), \qquad \text{MFCC} = \text{DCT-II}(S)[:, :13]
```

### Worked example

Settings: $f_s = 16{,}000$, $L = 16{,}000$ (1 s), $W = 400$, $H = 160$, $N_{\text{fft}} = 512$, $M = 80$.

1. **Frames:** $F = 1 + \lfloor (16{,}000 - 400)/160 \rfloor = 1 + \lfloor 15{,}600 / 160 \rfloor = 1 + 97 = 98$. Frame shape $98 \times 400$.
2. **Bins:** $512 / 2 + 1 = 257$ bins, spaced $16{,}000 / 512 = 31.25$ Hz. Power spectrogram shape $98 \times 257$.
3. **A 2 kHz tone** lands at bin $2{,}000 / 31.25 = 64$ exactly.
4. **Mel range:** $m(8{,}000) = 2595 \log_{10}(1 + 8{,}000/700) = 2595 \log_{10}(12.4286) = 2595 \times 1.09442 = 2840.0$ mel. 82 edges from 0 to 2840.0 are $2840.0 / 81 = 35.06$ mel apart.
5. **Where 2 kHz falls:** $m(2{,}000) = 2595 \log_{10}(3.8571) = 2595 \times 0.58627 = 1521.4$ mel. Edges 43 and 44 are at $43 \times 35.06 = 1507.7$ and $44 \times 35.06 = 1542.7$ mel, which convert to $e_{43} = 1967.4$ Hz and $e_{44} = 2051.7$ Hz.
6. **Weights for bin 64 (2,000 Hz):** filter 42 (edges 42, 43, 44) is on its falling side: $(2051.7 - 2000)/(2051.7 - 1967.4) = 51.7 / 84.3 = 0.613$. Filter 43 (edges 43, 44, 45) is on its rising side: $(2000 - 1967.4)/84.3 = 32.6 / 84.3 = 0.387$. The two weights sum to 1, because adjacent triangles overlap exactly.
7. **Shapes:** $B$ is $80 \times 257$; $P B^\top$ is $(98 \times 257)(257 \times 80) = 98 \times 80$; MFCCs are $98 \times 13$.
8. **Filter widths:** filter 0 spans 0 to 45 Hz, so with bins 31.25 Hz apart it contains a single FFT bin (31.25 Hz). Filter 79 spans 7,475 to 8,000 Hz (525 Hz). Equal steps in mel are very unequal in hertz.

**Time versus frequency resolution.** Bin spacing is $f_s / N_{\text{fft}}$, and each frame spans $W / f_s$ seconds. A longer window resolves nearby pitches but blurs fast events; a shorter one does the opposite. 25 ms / 10 ms is the speech compromise.

## 4. Implementation

### Build it from scratch

All of it is NumPy. First, framing. Instead of a Python loop, build a $F \times W$ index matrix: row $t$ is $tH, tH+1, \dots, tH+W-1$, and fancy indexing pulls every frame at once.

```python
def frame(x, win=WIN, hop=HOP):
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]           # [n_frames, win]
    return x[idx]
```

*Predict:* a 0.5 s clip at 16 kHz with the same window and hop gives how many frames?

Window and FFT. `np.fft.rfft` returns only the non-negative frequencies, and `n=N_FFT` zero-pads each 400-sample frame to 512.

```python
def hann(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)              # periodic Hann (as torch/librosa)

def power_spectrum(frames, n_fft=N_FFT):
    spec = np.fft.rfft(frames * hann(frames.shape[1]), n=n_fft)           # [n_frames, n_fft//2 + 1]
    return np.abs(spec) ** 2
```

*Predict:* zero-padding to 512 makes the bins closer together (31.25 Hz instead of 40 Hz). Does it let you separate two tones 35 Hz apart? (No: padding interpolates the spectrum; true resolution is set by the 25 ms window.)

The mel conversion and the filterbank. Each row of `fb` is one triangle evaluated at every FFT bin's frequency.

```python
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
```

*Predict:* what is the shape of `fb` for 80 mels and `n_fft=512`, and roughly how many nonzero entries does row 79 have? (Script: $80 \times 257$, 17.)

Log-mel is one matmul and a log; MFCC is a DCT-II written as another matmul with a cosine basis.

```python
def log_mel(power, fb, eps=1e-10):
    return np.log10(np.maximum(power @ fb.T, eps))                      # [n_frames, n_mels]

def dct_ii(x, n_coeffs=13):
    N = x.shape[-1]
    n = np.arange(N)
    basis = np.cos(np.pi * np.arange(n_coeffs)[:, None] * (2 * n + 1) / (2 * N))   # [n_coeffs, N]
    return x @ basis.T
```

*Predict:* in a silent frame, what value does every log-mel entry take? (The floor: $\log_{10}(10^{-10}) = -10$. That's why the floor value matters.)

### Use the library

The same pipeline in one call each:

```python
import torchaudio, librosa

mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=400,
                                           hop_length=160, n_mels=80)(waveform)  # [channels, 80, frames]
log_mel = torch.log10(mel.clamp(min=1e-10))

S = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=400, hop_length=160, n_mels=80)
mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=13)
```

Note the conventions that differ from our script and from each other: libraries put time on the *last* axis ($80 \times F$, not $F \times 80$); librosa's default mel scale is the Slaney variant (linear below 1 kHz) with area-normalized filters, while torchaudio's default is the HTK formula used here; `power_to_db` gives $10 \log_{10}$ relative to a reference. Whisper computes 25 ms / 10 ms frames at 16 kHz, applies a mel filterbank (80 bands in most sizes, 128 in large-v3), takes $\log_{10}$, clips to 8 below the maximum, and rescales. If you feed a pretrained model, use its own feature extractor (for example the Hugging Face `WhisperFeatureExtractor`), never a re-implementation.

### Run it

```bash
cd guide
uv run code/29-speech-and-audio/mel_features.py
```

Key output:

```
frames: 1 + (16000 - 400) // 160 = 98  -> shape (98, 400)
power spectrum shape (98, 257); bin spacing 31.25 Hz
frame 70 (t = 0.70 s): peak bin 64 -> 2000.0 Hz (true 2000 Hz)
filterbank shape (80, 257); filter 0 spans 0-45 Hz (45 Hz wide), filter 79 spans 7475-8000 Hz (525 Hz wide)
log-mel shape (98, 80); MFCC shape (98, 13)
  frame 25 (t = 0.263 s): mel 45, centre   2228 Hz, true   2195 Hz
silent frame 95: max log-mel -10.0 (floor log10(1e-10) = -10)
```

Full script: [`code/29-speech-and-audio/mel_features.py`](../../code/29-speech-and-audio/mel_features.py).

## 5. Engineering

**Typical settings.**

| Task | Sample rate | Window / hop | Features |
|---|---|---|---|
| Speech recognition | 16 kHz | 25 ms / 10 ms | 80 (sometimes 128) log-mels |
| Speech synthesis acoustic models | 22.05 or 24 kHz | about 46 ms / 11 ms | 80 to 100 log-mels |
| General audio classification | 16 to 32 kHz | 25 ms / 10 ms | 64 to 128 log-mels |
| Speaker verification, small devices | 16 kHz | 25 ms / 10 ms | log-mels or MFCC-13 to 40 |

**Sample rate is part of the model.** A model trained at 16 kHz fed 44.1 kHz audio sees every frequency scaled wrong. Resample at the input boundary, with a proper anti-aliasing filter (`torchaudio.functional.resample`, `librosa.resample`), and assert the rate in code.

**Normalization must match training.** Per-utterance mean/variance normalization, global statistics, or Whisper-style clipping and rescaling all produce different numbers. A mismatch can silently degrade accuracy. Ship the feature extractor with the model and test that features match on a golden file.

**Padding and length.** Whisper pads or trims every input to 30 s (3,000 frames). Streaming models need features computed incrementally with the same framing as offline. Zero-padding produces frames at the log floor; make sure the model saw the same during training, or pass an attention mask.

**Compute.** The front end is cheap next to the model (an FFT per 10 ms frame), but on small devices it matters: precompute the filterbank once, use real FFTs, and run in float32.

**Augmentation.** SpecAugment (masking random time and frequency bands of the log-mel) is a standard, cheap regularizer for speech models, applied to features during training only.

### Common mistakes

- Feeding audio at the wrong sample rate, or resampling with naive decimation that aliases.
- Mixing natural log, $\log_{10}$, and decibels between training and serving.
- Transposing time and mel axes ($F \times 80$ vs $80 \times F$) when moving between libraries.
- Too many mel filters for the FFT size, leaving low filters covering one or zero bins (librosa warns about empty filters).
- Re-implementing a pretrained model's feature extractor "close enough" instead of using its own.
- Forgetting the window, which smears every tone across neighbouring bins.

## 6. Knowledge check

<!-- quiz:spectrograms-mel-features -->
**[Take the spectrograms and mel features quiz](../../quizzes/spectrograms-mel-features.md)**
<!-- /quiz -->

**Practice exercise.** A 3-second clip at 16 kHz uses a 25 ms window, 10 ms hop, $N_{\text{fft}} = 400$, and 80 mels. Give the number of samples, frames, FFT bins, bin spacing in hertz, and the shapes of the power spectrogram, filterbank, and log-mel output.

<details>
<summary>Solution</summary>

Samples: $3 \times 16{,}000 = 48{,}000$. Window $0.025 \times 16{,}000 = 400$ samples; hop $0.010 \times 16{,}000 = 160$.
Frames: $1 + \lfloor (48{,}000 - 400) / 160 \rfloor = 1 + \lfloor 297.5 \rfloor = 1 + 297 = 298$.
Bins: $400/2 + 1 = 201$, spacing $16{,}000 / 400 = 40$ Hz.
Power spectrogram $298 \times 201$; filterbank $80 \times 201$; log-mel $298 \times 80$.
(Libraries that center frames with padding give $1 + \lfloor 48{,}000 / 160 \rfloor = 301$ frames instead. Always check the `center` setting.)

</details>

**Implementation challenge.** Add delta features: for each log-mel frame $t$, compute $\Delta_t = (S_{t+1} - S_{t-1}) / 2$ (edge-replicate the ends), then stack $[S, \Delta]$ into a $F \times 160$ matrix. Check on the chirp that the delta is positive just above the chirp's current band and negative just below it.

<details>
<summary>Solution sketch</summary>

`P = np.pad(S, ((1, 1), (0, 0)), mode="edge")`, then `delta = (P[2:] - P[:-2]) / 2`, then `np.hstack([S, delta])`. As the chirp rises, the band just above it gets louder over time (positive delta) and the band it is leaving gets quieter (negative delta). Deltas were standard with MFCCs; neural models mostly learn temporal change themselves from stacked frames, but deltas still help small models.

</details>

## Summary

- Speech front ends frame the waveform (25 ms window, 10 ms hop at 16 kHz), apply a Hann window, and take an FFT per frame to get a power spectrogram.
- The mel filterbank is an $M \times (N_{\text{fft}}/2+1)$ matrix of triangles equally spaced in mel; one matrix multiply turns 257 bins into 80 bands.
- Log compresses dynamic range; MFCCs add a DCT and keep about 13 coefficients, now mostly replaced by log-mels for neural models.
- Window length trades time resolution against frequency resolution.
- Sample rate, log base, normalization, axis order, and padding must match what the model was trained on.

**Next:** [Speech recognition: CTC, transducers, and attention](02-speech-recognition-asr.md)

**Related:** [Convolutional networks](../13-deep-architectures/01-convolutional-networks.md) · [Recurrent networks](../13-deep-architectures/02-recurrent-networks.md) · [Transformer architecture](../14-transformers/02-transformer-architecture.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
**Go deeper, code-first** ([AI Engineering from Scratch](https://aiengineeringfromscratch.com)): [The Fourier Transform](https://aiengineeringfromscratch.com/lesson?path=phases/01-math-foundations/20-fourier-transform) · [Audio Fundamentals — Waveforms, Sampling, Fourier Transform](https://aiengineeringfromscratch.com/lesson?path=phases/06-speech-and-audio/01-audio-fundamentals) · [Audio Classification — From k-NN on MFCCs to AST and BEATs](https://aiengineeringfromscratch.com/lesson?path=phases/06-speech-and-audio/03-audio-classification)

*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Spectrograms, Mel Scale & Audio Features](https://aiengineeringfromscratch.com/lesson?path=phases/06-speech-and-audio/02-spectrograms-mel-features).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>Walk me through how raw audio becomes the input to a speech model.</strong></summary>

Start with a mono waveform at the model's sample rate, usually 16 kHz. Cut it into overlapping frames, typically 25 milliseconds every 10 milliseconds, so 400 samples with a hop of 160. Multiply each frame by a Hann window so the edges taper to zero and the FFT doesn't smear energy. Take a real FFT and the squared magnitude, giving a power spectrum per frame. Multiply by a mel filterbank, a matrix of triangular filters equally spaced on the mel scale, which turns roughly 200 to 257 frequency bins into 80 bands, fine at low frequencies and coarse at high ones. Take the log, with a small floor, to compress dynamic range. Finally apply the same normalization the model was trained with. For a one-second clip that's a 98 by 80 matrix.

</details>

<details>
<summary><strong>How do you choose the window length, and what does zero-padding the FFT buy you?</strong></summary>

Window length sets the trade between time and frequency resolution. The true frequency resolution is about the sample rate divided by the window length, so 25 milliseconds gives roughly 40 hertz; a 100-millisecond window would resolve 10 hertz but blur anything faster than 100 milliseconds, like consonant onsets. Speech uses 25 milliseconds because speech is roughly stable over that span, and a 10-millisecond hop tracks how fast it changes. Music analysis often uses longer windows for pitch detail; drum or click detection uses shorter ones. Zero-padding the frame to a larger FFT size only interpolates the spectrum onto a finer grid. It makes peaks look smoother and can align bins with filter edges, but it cannot separate two tones the window itself can't resolve.

</details>

<details>
<summary><strong>A speech model scores well offline but badly in production on the same kind of audio. Where do you look first?</strong></summary>

The front end, before the model. First sample rate: production audio at 8 kHz telephony or 44.1 or 48 kHz from a browser fed into a 16 kHz model, or resampled without anti-aliasing. Second, the feature extractor itself: a re-implementation that uses natural log instead of log10, decibels instead of log power, a different mel scale variant, different normalization, or time and mel axes transposed. Third, padding and length: training clips padded to 30 seconds but production chunks of 2 seconds, or streaming code that frames differently at chunk boundaries. Fourth, channel and gain: stereo not downmixed, clipping, or very low levels. I'd dump features for the same file from the offline and production paths and diff them numerically, then add that golden-file comparison as a test.

</details>

<details>
<summary><strong>How many feature frames does a 30-second clip produce at 16 kHz with a 10 ms hop, and what does that mean for a transformer encoder?</strong></summary>

30 seconds at 100 frames per second is about 3,000 frames; with a 400-sample window and no centering it's 1 plus 479,600 over 160, which is 2,998. Self-attention cost grows with the square of sequence length, so 3,000 frames is expensive per layer. That's why speech encoders downsample early: Whisper's convolutional stem has a stride of 2, so the transformer sees 1,500 positions for 30 seconds; Conformer-style encoders often subsample by 4 or 8. With 80 mel bands, the input tensor is 80 by 3,000 per clip. Downsampling to 40 or 80 milliseconds per position still leaves enough resolution for characters or subword tokens.

</details>
