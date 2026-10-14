---
title: Speech and audio
summary: "Turn sound into model input and text: spectrograms, mel filterbanks, MFCCs, CTC, transducers, attention decoders, and word error rate."
skill: deep-learning
---

# Module 29: Speech and audio

Audio arrives as tens of thousands of samples per second. Speech models first turn it into a log-mel spectrogram, a compact time-by-frequency image, then align those frames to a much shorter sequence of text tokens. This module builds both halves from scratch in NumPy and connects them to the libraries and pretrained models you would use in practice.

| Stage | What happens | Failure mode |
|---|---|---|
| Resample | bring audio to the model's rate (often 16 kHz) | wrong rate, aliasing |
| Frame and window | 25 ms slices every 10 ms, tapered | leakage, wrong framing |
| FFT and mel | power per frequency, pooled into mel bands | mismatched filterbank or log |
| Encode | convolution and transformer layers | too much downsampling for CTC |
| Align and decode | CTC, transducer, or attention decoder | dropped double letters, hallucination |
| Evaluate | word error rate after normalization | incomparable numbers |

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [Spectrograms and mel features](01-spectrograms-mel-features.md) | compute frames, STFT, mel filterbank, log-mel, and MFCC with correct shapes, and avoid front-end mismatches |
| 2 | [Speech recognition: CTC, transducers, and attention](02-speech-recognition-asr.md) | decode CTC output, compute CTC loss with the forward algorithm, measure WER, and choose an ASR architecture |
