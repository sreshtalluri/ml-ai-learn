---
title: "Speech recognition: CTC, transducers, and attention"
summary: Explain how speech recognizers align audio frames to text (CTC, RNN-T, attention encoder-decoder), decode CTC output greedily and with prefix beam search, compute the CTC loss with the forward algorithm, and measure word error rate.
skill: deep-learning
minutes: 45
prerequisites: [spectrograms-mel-features, recurrent-networks, transformer-architecture]
related: [decoding, self-attention, llm-evaluation]
adapted_from: [phases/06-speech-and-audio/04-speech-recognition-asr]
---

# Speech recognition: CTC, transducers, and attention

**Mental model.** A speech recognizer listens to 100 slices of sound per second but writes maybe 15 letters per second. Something has to decide which slices belong to which letter. CTC lets the model say "letter, letter, nothing, letter" for every slice and then squeezes out the repeats and the nothings. Transducers and attention models solve the same alignment problem in different ways.

**You will learn to**

- State the alignment problem in speech recognition and how CTC, RNN-T, and attention encoder-decoders each handle it.
- Decode CTC output with the collapse rule (greedy) and with prefix beam search, and explain why they can disagree.
- Compute the CTC probability of a transcript with the forward algorithm, and check it against brute force on a tiny example.
- Compute word error rate with edit distance, including the split into substitutions, deletions, and insertions.
- Choose an ASR architecture for offline, streaming, and on-device use.

**Why it matters.** Speech is an input to search, assistants, meeting notes, call analytics, and voice agents built on LLMs. Most teams use a pretrained recognizer, but choosing one, streaming it, and debugging it requires knowing how it aligns audio to text and how its errors are measured. CTC is also a general idea: any time a long input sequence must produce a shorter output sequence without known alignment (handwriting, sign language, DNA basecalling), CTC is a candidate.

## 1. Intuition

**The alignment problem.** After the [front end](01-spectrograms-mel-features.md), a clip is a sequence of feature frames, 100 per second. An encoder (convolutions, then a transformer or Conformer stack) turns them into $T$ hidden vectors, often downsampled to one per 40 to 80 ms. The transcript is a shorter sequence of $U$ tokens (characters or subwords). We don't know which frames produced which token, and a word can take 150 ms or 1.5 s. Training data has only (audio, transcript) pairs.

**[CTC](../../glossary.md#ctc) (Connectionist Temporal Classification).** Add a special **blank** token meaning "nothing new here". The model outputs a probability distribution over tokens-plus-blank at *every* frame, independently. Any frame-level path maps to text by a collapse rule: **merge consecutive repeats, then delete blanks**. So `hh_e_ll_lloo` becomes "hello": the blank between the two `ll` groups is what keeps the double letter. Training maximizes the total probability of *all* paths that collapse to the correct transcript. Decoding is fast because frames are independent. The cost is that independence: CTC can't model that "their" and "there" depend on the surrounding words, so it is often paired with an external language model at decode time.

**RNN-T (transducer).** Add a small prediction network that reads the tokens emitted so far, and a joint network that combines it with the current audio frame to predict "next token, or move to the next frame". Output now depends on previous outputs, an internal language model of sorts, while still processing audio left to right. That makes transducers the standard choice for streaming and on-device recognition. Training is heavier: the loss sums over a 2D lattice of (frame, token) positions. Token-and-duration transducers (TDT) also predict how many frames to skip, which speeds up decoding.

**Attention encoder-decoder.** The encoder reads all the audio; a transformer decoder generates tokens one at a time, cross-attending to any part of the encoder output. This is the same autoregressive generation as an LLM (see [decoding](../15-llms/02-decoding.md)); Whisper is the best-known example. It gives strong offline accuracy and handles punctuation, casing, and translation naturally. It doesn't stream without extra engineering, latency grows with output length, and like any autoregressive generator it can **hallucinate**: produce fluent text that wasn't said, especially on silence, music, or noise.

**Measuring it: WER.** [Word error rate](../../glossary.md#word-error-rate) counts the minimum number of word substitutions, deletions, and insertions that turn the hypothesis into the reference, divided by the number of reference words. Lower is better, and it can exceed 1.0 because insertions are unbounded.

> [!IMPORTANT]
> **WER depends on text normalization.** "Turn on the lights." and "turn on the lights" differ by casing and punctuation. Published comparisons apply a normalizer (lowercase, strip punctuation, standardize numbers and spellings) before computing WER. Comparing your system's raw WER to a published normalized WER is meaningless.

## 2. Visualization

![Left: heatmap of synthetic frame posteriors for "hello world", 46 frames by the 9 tokens used (blank, h, e, l, o, space, w, r, d). Each letter is a bright block of three frames, separated by single blank frames, with orange dots marking the argmax at each frame. Right: heatmap of the CTC forward variables over the first 20 frames for the target "hello", whose extended label sequence is blank h blank e blank l blank l blank o blank. Bright cells form a diagonal band climbing from the bottom-left to the top-right, showing all the alignments still consistent with the transcript.](../../figures/speech-recognition-asr.png)

*Synthetic posteriors (no audio): what an acoustic encoder might output for "hello world". Right: the forward algorithm's table, in log10, restricted to the first 20 frames.*

On the left, the orange dots are greedy decoding: take the most likely token at each frame. Reading them in order gives `_hhh_eee_lll_lll_ooo_ …`, and the collapse rule turns that into "hello world". Note the blank between the two `lll` blocks: without it, the collapse would merge them into one `l`. On the right, each column is one frame and each row is a position in `_h_e_l_l_o_`. A bright cell means "a lot of probability flows through being at this label position at this frame". The diagonal band is the set of plausible alignments; the loss sums them all instead of picking one.

**Try it** (predict first, then edit `ctc_asr.py`):

1. In `synthetic_posteriors`, remove the line that inserts a blank between repeated letters. Predict the greedy output for "hello".
2. Lower `peak` from 0.8 to 0.3. Predict whether greedy and beam search still agree, and what kinds of errors appear.
3. In the two-frame example, change the probabilities to $P(\_) = 0.45$, $P(a) = 0.55$. Predict the greedy output and $P(\text{"a"})$.
4. Add the hypothesis "turn off the kitchen lights" to the WER list. Predict its WER and the S/D/I split.

## 3. The math

### Symbols

| Symbol | Meaning | Shape / unit |
|---|---|---|
| $x$ | encoder input (log-mel frames) | $T_{\text{in}} \times 80$ |
| $T$ | number of encoder output frames | count |
| $V$ | vocabulary size, not counting blank | count |
| $p_t(k)$ | probability of token $k$ at frame $t$ (softmax output) | $T \times (V+1)$ |
| $y$ | target transcript | length $U$ |
| $\pi$ | a frame-level path | length $T$ |
| $\mathcal{B}$ | collapse map: merge repeats, delete blanks | path → text |
| $y'$ | $y$ with blanks between and around tokens | length $S = 2U + 1$ |
| $\alpha_t(s)$ | probability of all path prefixes up to frame $t$ ending at position $s$ of $y'$ | $T \times S$ |
| $S, D, I, N$ | substitutions, deletions, insertions, reference word count | counts |

### CTC probability and loss

```math
P(y \mid x) = \sum_{\pi \,:\, \mathcal{B}(\pi) = y} \; \prod_{t=1}^{T} p_t(\pi_t), \qquad \mathcal{L}_{\text{CTC}} = -\ln P(y \mid x)
```

There are $(V+1)^T$ paths, far too many to enumerate. The forward algorithm sums them in $O(T \cdot S)$ time. With $y' = (\_, y_1, \_, y_2, \dots, y_U, \_)$:

```math
\alpha_t(s) = \Big(\alpha_{t-1}(s) + \alpha_{t-1}(s-1) + \mathbb{1}[\text{skip allowed}]\,\alpha_{t-1}(s-2)\Big)\, p_t(y'_s)
```

A path can stay at the same label, move one position forward, or skip one blank. The skip is allowed only when $y'_s$ is not blank and $y'_s \ne y'_{s-2}$: you can't jump over the blank between two identical letters, because that would merge them. It starts with $\alpha_1(1) = p_1(\_)$, $\alpha_1(2) = p_1(y_1)$, and ends with $P(y \mid x) = \alpha_T(S) + \alpha_T(S-1)$ (finish on the final blank or the final letter). A valid path needs $T \ge U + r$, where $r$ is the number of adjacent repeated letters in $y$, since each repeat needs a blank between.

### Worked example: T = 3, vocabulary {blank, a}, target "a"

Frame probabilities ($p_t(\_)$, $p_t(a)$): $(0.4, 0.6)$, $(0.5, 0.5)$, $(0.7, 0.3)$. Extended target $y' = (\_, a, \_)$, $S = 3$.

**Brute force.** Of the $2^3 = 8$ paths, six collapse to "a":
`__a` $= 0.4 \times 0.5 \times 0.3 = 0.06$; `_a_` $= 0.4 \times 0.5 \times 0.7 = 0.14$; `_aa` $= 0.4 \times 0.5 \times 0.3 = 0.06$; `a__` $= 0.6 \times 0.5 \times 0.7 = 0.21$; `aa_` $= 0.6 \times 0.5 \times 0.7 = 0.21$; `aaa` $= 0.6 \times 0.5 \times 0.3 = 0.09$.
Sum $= 0.06 + 0.14 + 0.06 + 0.21 + 0.21 + 0.09 = 0.77$. (`___` gives "" and `a_a` gives "aa".)

**Forward algorithm.**

1. $t = 1$: $\alpha_1 = (0.4, \; 0.6, \; 0)$.
2. $t = 2$, multiply by $(0.5, 0.5, 0.5)$: $\alpha_2(1) = 0.4 \times 0.5 = 0.2$; $\alpha_2(2) = (0.6 + 0.4) \times 0.5 = 0.5$; $\alpha_2(3) = (0 + 0.6) \times 0.5 = 0.3$.
3. $t = 3$, probabilities $(0.7, 0.3, 0.7)$ for $(\_, a, \_)$: $\alpha_3(1) = 0.2 \times 0.7 = 0.14$; $\alpha_3(2) = (0.5 + 0.2) \times 0.3 = 0.21$; $\alpha_3(3) = (0.3 + 0.5) \times 0.7 = 0.56$.
4. $P(\text{"a"}) = \alpha_3(3) + \alpha_3(2) = 0.56 + 0.21 = 0.77$. ✓
5. Loss $= -\ln 0.77 = 0.2614$.

### Greedy is not the most probable transcript

Two frames, each with $p(\_) = 0.6$, $p(a) = 0.4$. Greedy picks `__`, which collapses to "" with probability $0.6 \times 0.6 = 0.36$. But "a" collects three paths: `aa` $0.16$ + `a_` $0.24$ + `_a` $0.24$ $= 0.64$. The best *path* is not the best *transcript*. Prefix beam search fixes this by summing path probabilities per collapsed prefix.

### Word error rate

```math
\text{WER} = \frac{S + D + I}{N}
```

Reference "turn on the kitchen lights" ($N = 5$), hypothesis "turn the kitchen light on": "on" deleted after "turn" ($D = 1$), "lights" → "light" ($S = 1$), "on" inserted at the end ($I = 1$). $\text{WER} = 3/5 = 0.60$. The hypothesis "um so turn uh on the the kitchen lights please" contains every reference word but adds 5 insertions: $\text{WER} = 5/5 = 1.00$.

## 4. Implementation

### Build it from scratch

Start with the collapse rule and greedy decoding. `prev` tracks the last frame's token, so repeats merge, and blanks are dropped but still reset `prev`, which is what lets "ll" survive.

```python
def collapse(ids):
    out, prev = [], None
    for i in ids:
        if i != prev and i != BLANK:
            out.append(i)
        prev = i
    return out

def greedy(probs):
    return "".join(VOCAB[i] for i in collapse(probs.argmax(1)))
```

*Predict:* what does `collapse` return for the path `a a _ a b b`?

The forward algorithm. Build the extended label sequence, then fill the $T \times S$ table using stay, step, and skip.

```python
def ctc_prob(probs, target):
    ext = [BLANK]
    for c in target:
        ext += [c, BLANK]                             # "ab" -> _ a _ b _ ; length 2U + 1
    S, T = len(ext), len(probs)
    alpha = np.zeros((T, S))
    alpha[0, 0] = probs[0, ext[0]]
    alpha[0, 1] = probs[0, ext[1]] if S > 1 else 0.0
    for t in range(1, T):
        for s in range(S):
            a = alpha[t - 1, s] + (alpha[t - 1, s - 1] if s >= 1 else 0.0)
            if s >= 2 and ext[s] != BLANK and ext[s] != ext[s - 2]:
                a += alpha[t - 1, s - 2]              # skip the blank between different letters
            alpha[t, s] = a * probs[t, ext[s]]
    return alpha[-1, -1] + (alpha[-1, -2] if S > 1 else 0.0), alpha
```

*Predict:* for the target "abb", which skip transitions are forbidden? (The jump from the first `b` over the blank to the second `b`.)

Real implementations work in log space (sums become log-sum-exp) because products of hundreds of probabilities underflow. The script checks the forward algorithm against brute-force enumeration on a random 6-frame case, so you can trust it.

Prefix beam search keeps, for each candidate text prefix, two probabilities: ending in blank and ending in a non-blank. The split is needed to decide whether the next identical letter extends the prefix or merges with it.

```python
def prefix_beam(probs, beam=8):
    beams = {(): (1.0, 0.0)}                        # prefix -> (P ending in blank, P ending in non-blank)
    for p in probs:
        nxt = defaultdict(lambda: [0.0, 0.0])
        for prefix, (pb, pnb) in beams.items():
            nxt[prefix][0] += (pb + pnb) * p[BLANK]                    # emit blank: prefix unchanged
            for c in range(1, len(p)):
                if prefix and prefix[-1] == c:
                    nxt[prefix][1] += pnb * p[c]                       # repeat without blank: merges
                    nxt[prefix + (c,)][1] += pb * p[c]                 # blank in between: new letter
                else:
                    nxt[prefix + (c,)][1] += (pb + pnb) * p[c]
        beams = dict(sorted(nxt.items(), key=lambda kv: -sum(kv[1]))[:beam])
    best = max(beams.items(), key=lambda kv: sum(kv[1]))
    return "".join(VOCAB[i] for i in best[0]), sum(best[1])
```

*Predict:* on the two-frame example ($p(\_) = 0.6$, $p(a) = 0.4$), what does `prefix_beam` return? (Script: "a" with 0.64.) This is where an external language model plugs in: multiply each extension by $P_{\text{LM}}(c \mid \text{prefix})^{\lambda}$ when a word completes.

Word error rate is edit distance over words; the script also backtraces the table to count S, D, and I separately.

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0], d[0, :] = np.arange(len(r) + 1), np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
```

*Predict:* "the cat sat" versus "the cat sat down". WER? ($1/3 = 0.33$, one insertion.)

### Use the library

Training with CTC in PyTorch is one loss call; inputs are log-probabilities shaped time-first.

```python
import torch.nn.functional as F
loss = F.ctc_loss(log_probs,        # [T, batch, V+1], from log_softmax
                  targets,          # concatenated target ids
                  input_lengths, target_lengths, blank=0)
```

For inference, use a pretrained model:

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo")
print(asr("clip.wav", return_timestamps=True)["text"])   # long audio is chunked into 30 s windows
```

Other common choices: `faster-whisper` (CTranslate2 backend, faster CPU/GPU inference of Whisper models), NVIDIA NeMo (Parakeet and Canary families: transducer, CTC, and attention models; strong streaming support), and `pyctcdecode` for CTC beam search with an n-gram language model. Compute WER with `jiwer` after applying the same text normalizer to reference and hypothesis (Whisper ships an `EnglishTextNormalizer`). Check the Hugging Face Open ASR Leaderboard for current, normalized results on public test sets instead of quoting numbers from memory.

### Run it

```bash
cd guide
uv run code/29-speech-and-audio/ctc_asr.py
```

Key output:

```
  forward algorithm P('a') = 0.770; brute force = 0.770; CTC loss = -ln P = 0.2614
  random check, T = 6, V = 4, target 'abb': forward 0.000370 vs brute force 0.000370
  P('a') = aa + a_ + _a = 0.16 + 0.24 + 0.24 = 0.64
  greedy -> ''; prefix beam -> 'a' with p = 0.64
  posteriors shape (46, 28) (frames x vocab)
  greedy: 'hello world'
  WER 0.60  S=1 D=1 I=1  'turn the kitchen light on'
  WER 1.00  S=0 D=0 I=5  'um so turn uh on the the kitchen lights please'
```

Full script: [`code/29-speech-and-audio/ctc_asr.py`](../../code/29-speech-and-audio/ctc_asr.py).

## 5. Engineering

**Choosing an architecture.**

| Need | Usual choice | Why |
|---|---|---|
| Best offline accuracy, punctuation, many languages | attention encoder-decoder (Whisper-style) or large transducer | full-context decoder, rich output |
| Live captions, voice agents, low latency | streaming transducer or chunked CTC | emits tokens as audio arrives |
| On-device, small footprint | small transducer or CTC with an n-gram LM | compact, no autoregressive decoder |
| Domain vocabulary (drug names, part numbers) | CTC/transducer with LM fusion or contextual biasing; or fine-tuning | boost rare words without retraining everything |

**Long audio.** Whisper-style models take 30 s windows. Split long audio with voice activity detection (VAD) at pauses, transcribe chunks (in parallel batches), and stitch with overlap. Splitting mid-word causes errors at the seams.

**Hallucination.** Autoregressive decoders can invent text on silence or music, or loop on a phrase. Mitigations: VAD to skip non-speech, compression-ratio and log-probability thresholds to reject degenerate outputs, temperature fallback, and not conditioning on the previous chunk's text when it's wrong.

**Language and domain.** Force the language when you know it; automatic language ID can misfire on short or noisy clips. Accented speech, telephone audio (8 kHz), far-field microphones, and overlapping speakers all raise WER sharply; evaluate on audio from your real channel, not just clean benchmarks.

**Evaluation.** Report WER on a held-out set from your domain with a fixed normalizer, plus a breakdown by speaker group, channel, and noise condition. For voice agents, also measure latency (time to first word, time to final transcript) and the effect of recognition errors on the downstream task, since the LLM may be robust to some errors and not others.

**Privacy.** Voice is biometric and often contains personal data. Decide what audio and transcripts you store, for how long, and whether processing can stay on device.

### Common mistakes

- Comparing WER numbers computed with different normalizers or on different test sets.
- Feeding 44.1 kHz or 8 kHz audio to a 16 kHz model without resampling.
- Forgetting that CTC needs $T \ge U + r$: downsampling the encoder too aggressively makes long or repetitive transcripts impossible to emit.
- Naive beam search that merges paths by collapsed text without tracking blank versus non-blank, which breaks double letters.
- Sending hours of audio to a 30-second-window model without chunking and VAD.
- Trusting confident-looking output on silence; attention decoders can produce fluent text from noise.

## 6. Knowledge check

<!-- quiz:speech-recognition-asr -->
**[Take the speech recognition quiz](../../quizzes/speech-recognition-asr.md)**
<!-- /quiz -->

**Practice exercise.** Two frames, vocabulary {blank, a, b}, probabilities per frame $(p_\_, p_a, p_b)$: frame 1 $(0.5, 0.3, 0.2)$, frame 2 $(0.4, 0.2, 0.4)$. Compute $P(\text{"a"})$, $P(\text{"b"})$, $P(\text{"ab"})$, and $P(\text{""})$, and give the greedy decode.

<details>
<summary>Solution</summary>

Paths of length 2:
"": `__` $= 0.5 \times 0.4 = 0.20$.
"a": `aa` $= 0.3 \times 0.2 = 0.06$; `a_` $= 0.3 \times 0.4 = 0.12$; `_a` $= 0.5 \times 0.2 = 0.10$. Total $0.28$.
"b": `bb` $= 0.2 \times 0.4 = 0.08$; `b_` $= 0.2 \times 0.4 = 0.08$; `_b` $= 0.5 \times 0.4 = 0.20$. Total $0.36$.
"ab": `ab` $= 0.3 \times 0.4 = 0.12$. "ba": `ba` $= 0.2 \times 0.2 = 0.04$.
Check: $0.20 + 0.28 + 0.36 + 0.12 + 0.04 = 1.00$. ✓

Greedy: frame 1 argmax is blank (0.5); frame 2 is a tie between blank and b (0.4). Breaking the tie toward blank gives ""; toward b gives "b". The most probable transcript is "b" (0.36), which beats "" (0.20) either way.

</details>

**Implementation challenge.** Convert `ctc_prob` to log space using `np.logaddexp`, and verify it matches the probability-space version on the "hello world" posteriors. Then compute the loss for a 200-frame synthetic utterance where the probability-space version underflows to 0.

<details>
<summary>Solution sketch</summary>

Store `la = np.full((T, S), -np.inf)`, initialize with `np.log` of the first-frame probabilities, and replace each sum with `np.logaddexp` over the two or three predecessors, then add `np.log(probs[t, ext[s]])`. The final log-probability is `np.logaddexp(la[-1, -1], la[-1, -2])`. On "hello world" it should equal `np.log` of the original to about 1e-12. For long utterances, the product of hundreds of probabilities around 0.1 underflows float64 near $10^{-308}$, while the log version stays finite. This is exactly what `F.ctc_loss` does internally.

</details>

## Summary

- Speech recognition must align many audio frames to fewer tokens without knowing the alignment.
- CTC predicts a token or blank per frame independently, collapses repeats and blanks, and trains on the sum over all valid paths, computed by the forward algorithm in O(T·S).
- Greedy CTC decoding picks the best path, not the best transcript; prefix beam search sums paths per prefix and is where language models plug in.
- Transducers add a prediction network for streaming with output dependencies; attention encoder-decoders like Whisper give strong offline quality but don't stream natively and can hallucinate.
- WER = (S + D + I) / N, computed after a fixed text normalization; it can exceed 1.

**Next:** [Gradient clipping and mixed precision](../27-training-at-scale/01-gradient-clipping-amp.md)

**Related:** [Spectrograms and mel features](01-spectrograms-mel-features.md) · [Recurrent networks](../13-deep-architectures/02-recurrent-networks.md) · [Transformer architecture](../14-transformers/02-transformer-architecture.md) · [LLM evaluation](../17-llm-evaluation/01-llm-evaluation.md)

<!-- deeper: generated from guide/upstream/map.json by web/scripts/gen-upstream.mjs; edit the map, then run npm run gen:upstream -->
**Go deeper, code-first** ([AI Engineering from Scratch](https://aiengineeringfromscratch.com)): [Whisper — Architecture & Fine-Tuning](https://aiengineeringfromscratch.com/lesson?path=phases/06-speech-and-audio/05-whisper-architecture-finetuning) · [Text-to-Speech (TTS) — From Tacotron to F5 and Kokoro](https://aiengineeringfromscratch.com/lesson?path=phases/06-speech-and-audio/07-text-to-speech) · [Audio Evaluation — WER, MOS, UTMOS, MMAU, FAD, and the Open Leaderboards](https://aiengineeringfromscratch.com/lesson?path=phases/06-speech-and-audio/17-audio-evaluation-metrics)

*Parts of this lesson are adapted from [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) by Rohit Ghumare (MIT license, [notice](../../upstream/LICENSE-ai-engineering-from-scratch.txt)): [Speech Recognition (ASR) — CTC, RNN-T, Attention](https://aiengineeringfromscratch.com/lesson?path=phases/06-speech-and-audio/04-speech-recognition-asr).*
<!-- /deeper -->

## Interview angle

<details>
<summary><strong>Explain CTC: what problem does it solve and how does the blank token work?</strong></summary>

Speech gives many more frames than output characters, and training data has no frame-level alignment. CTC lets the network output a distribution over characters plus a blank at every frame, independently. A frame-level path maps to text by merging consecutive repeats and then removing blanks, so "hh_e_ll_lo" becomes "hello". The blank does two jobs: it fills frames where nothing new is said, and it separates genuine double letters, since "ll" without a blank between would merge into one "l". Training maximizes the total probability of all paths that collapse to the transcript, summed efficiently with a forward algorithm over the transcript interleaved with blanks, in time proportional to frames times label length. The main weakness is the conditional independence between frames, so CTC is often paired with a language model during beam search.

</details>

<details>
<summary><strong>CTC, RNN-T, or attention encoder-decoder for a live captioning product? Why?</strong></summary>

I'd lean toward a streaming transducer. Live captions need words on screen within a few hundred milliseconds, so the model must emit as audio arrives. A transducer processes audio left to right with limited lookahead, and its prediction network conditions on previous tokens, so it captures some language context that pure CTC lacks. CTC is the simpler, faster alternative: chunked encoders with CTC stream well and decode in parallel, and an n-gram or neural language model in beam search recovers much of the accuracy. An attention encoder-decoder like Whisper attends over a whole 30-second window and generates autoregressively, which gives great offline accuracy but needs chunking tricks and re-decoding to stream, adding latency and instability at chunk edges. I'd still measure: WER on our audio, latency to first and final word, and cost per hour.

</details>

<details>
<summary><strong>Your Whisper-based transcription inserts sentences nobody said during long silences. How do you fix it?</strong></summary>

That's decoder hallucination: an autoregressive model asked to produce text from near-silence generates fluent, plausible text, sometimes repeating earlier phrases or stock phrases from training data. First, stop feeding it silence: run voice activity detection and only transcribe speech regions. Second, use the decoder's own signals: reject or retry segments with a low average log-probability, a high no-speech probability, or a high compression ratio, which flags repetition loops. Third, check the decoding setup: conditioning on the previous segment's text can propagate a hallucination, so disable it after a bad segment, and use temperature fallback when greedy output looks degenerate. Then measure insertion rate separately from total WER on a test set with real silences, since insertions are exactly what this failure produces.

</details>

<details>
<summary><strong>A reference has 20 words. The hypothesis has 2 substitutions, 1 deletion, and 3 insertions. What is the WER, and why can WER exceed 100 percent?</strong></summary>

WER is substitutions plus deletions plus insertions over reference words: 2 plus 1 plus 3 is 6, over 20, so 0.30 or 30 percent. It can exceed 100 percent because insertions aren't bounded by the reference length. If the system outputs 25 extra words of hallucinated text on a 20-word reference while getting the 20 right, that's 25 over 20, or 125 percent. That's also why I look at the S, D, I breakdown rather than just the total: many deletions suggest VAD or chunking is cutting speech, many insertions suggest hallucination or noise being transcribed, and substitutions point at acoustic or vocabulary problems. All of it assumes reference and hypothesis went through the same normalizer.

</details>
