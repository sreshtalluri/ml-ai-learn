<!-- GENERATED from spectrograms-mel-features.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Spectrograms and mel features

Covers the lesson [Spectrograms and mel features](../lessons/29-speech-and-audio/01-spectrograms-mel-features.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/spectrograms-mel-features/) grades these interactively and tracks a review queue.

## 1. Calculation (medium)

A 2-second clip at 16 kHz is framed with a 400-sample window and a 160-sample hop (no padding). How many frames?

<details>
<summary>Answer</summary>

**198**

$L = 32{,}000$. $1 + \lfloor (32{,}000 - 400)/160 \rfloor = 1 + \lfloor 197.5 \rfloor = 1 + 197 = 198$.

</details>

## 2. Calculation (easy)

With a 16 kHz sample rate and a 512-point FFT, how far apart (in Hz) are adjacent frequency bins?

<details>
<summary>Answer</summary>

**31.25** (within ±0.01)

Bin spacing is $f_s / N_{\text{fft}} = 16{,}000 / 512 = 31.25$ Hz.

</details>

## 3. Calculation (medium)

Using $m = 2595 \log_{10}(1 + f/700)$, what is 700 Hz in mels? (1 decimal)

<details>
<summary>Answer</summary>

**781.2** (within ±0.2)

$1 + 700/700 = 2$. $\log_{10} 2 = 0.30103$. $2595 \times 0.30103 = 781.2$.

</details>

## 4. Multiple choice (easy)

Why is each frame multiplied by a Hann window before the FFT?

- **A.** To make the frame louder
- **B.** To taper the edges to zero so the cut doesn't look like a jump, which reduces spectral leakage
- **C.** To convert hertz to mels
- **D.** To remove silence

<details>
<summary>Answer</summary>

**B.** To taper the edges to zero so the cut doesn't look like a jump, which reduces spectral leakage

A hard cut adds broadband energy that smears a pure tone across many bins. Tapering reduces it.

- **A:** The window reduces amplitude near the edges; it doesn't amplify.
- **B:** Correct.
- **C:** That's the mel filterbank's job.
- **D:** Silence removal is voice activity detection.

</details>

## 5. Multiple choice (medium)

A power spectrogram has shape 98 x 257 (frames x bins). The mel filterbank has 80 filters. What is the filterbank's shape and the log-mel output's shape?

- **A.** Filterbank 257 x 98, output 80 x 80
- **B.** Filterbank 80 x 257, output 98 x 80
- **C.** Filterbank 80 x 98, output 257 x 80
- **D.** Filterbank 98 x 80, output 98 x 257

<details>
<summary>Answer</summary>

**B.** Filterbank 80 x 257, output 98 x 80

Each filter is a weight per FFT bin, so 80 x 257. Then (98 x 257)(257 x 80) = 98 x 80.

- **A:** The filterbank maps bins to mels; it doesn't depend on the number of frames.
- **B:** Correct.
- **C:** The filterbank acts on the frequency axis, not the time axis.
- **D:** The output must have 80 mel columns, not 257 bins.

</details>

## 6. Multiple choice (medium)

You lengthen the analysis window from 25 ms to 100 ms. What happens?

- **A.** Better time resolution and better frequency resolution
- **B.** Better frequency resolution, worse time resolution (fast events blur)
- **C.** Worse frequency resolution, better time resolution
- **D.** No change; only the FFT size matters

<details>
<summary>Answer</summary>

**B.** Better frequency resolution, worse time resolution (fast events blur)

Frequency resolution is about 1 / window duration; a long window averages over more time.

- **A:** The trade-off can't be avoided by changing window length.
- **B:** Correct.
- **C:** That's what a shorter window does.
- **D:** Zero-padding to a larger FFT only interpolates; true resolution comes from the window length.

</details>

## 7. Multiple choice (hard)

A speech model trained at 16 kHz performs badly in a browser app. The app sends 48 kHz audio straight into the model's feature extractor. What is happening?

- **A.** The model needs more training data
- **B.** Every frequency lands in the wrong bins and frames span a third of the intended time, so features don't match training
- **C.** 48 kHz has too much noise for any model
- **D.** The log base is wrong

<details>
<summary>Answer</summary>

**B.** Every frequency lands in the wrong bins and frames span a third of the intended time, so features don't match training

Sample rate is part of the model. Resample to 16 kHz with anti-aliasing at the input boundary.

- **A:** Data doesn't fix a front-end mismatch.
- **B:** Correct.
- **C:** Higher sample rate isn't noisier; it's just a different rate.
- **D:** The log may be fine; the rate is the clear mismatch here.

</details>

## 8. Multiple choice (medium)

How are MFCCs computed from a log-mel spectrogram, and why are they less common in modern ASR?

- **A.** By an inverse FFT; they are slower to compute
- **B.** By a DCT of each log-mel frame, keeping about 13 coefficients; neural networks learn well from log-mels directly, so the extra compression isn't needed
- **C.** By averaging mel bands; they lose timing
- **D.** By a second mel filterbank; they double the feature size

<details>
<summary>Answer</summary>

**B.** By a DCT of each log-mel frame, keeping about 13 coefficients; neural networks learn well from log-mels directly, so the extra compression isn't needed

The DCT decorrelates and compresses, which suited older Gaussian models. Neural nets don't need it.

- **A:** MFCCs use a DCT on log-mels, and they're cheap.
- **B:** Correct.
- **C:** MFCCs are per frame; timing is kept.
- **D:** MFCCs reduce the feature count, not double it.

</details>

## 9. Arrange in order (easy)

Put the front-end steps in order.

- Take the log
- Apply the mel filterbank
- FFT and take power
- Multiply each frame by a window
- Cut into overlapping frames
- Resample to the model's sample rate

<details>
<summary>Answer</summary>

1. Resample to the model's sample rate
2. Cut into overlapping frames
3. Multiply each frame by a window
4. FFT and take power
5. Apply the mel filterbank
6. Take the log

Rate first, then frame, window, FFT, mel pooling, and log compression.

</details>
