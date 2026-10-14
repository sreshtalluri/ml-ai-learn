<!-- GENERATED from speech-recognition-asr.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Speech recognition

Covers the lesson [Speech recognition](../lessons/29-speech-and-audio/02-speech-recognition-asr.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/speech-recognition-asr/) grades these interactively and tracks a review queue.

## 1. Fill in (easy)

Apply the CTC collapse rule (merge repeats, then delete blanks `_`) to the path `cc_aa_t_t`. What text results?

<details>
<summary>Answer</summary>

**catt**

`cc` → c, `_`, `aa` → a, `_`, `t`, `_`, `t`. The blank between the two t's keeps both: catt.

</details>

## 2. Multiple choice (easy)

Why does CTC need a blank token?

- **A.** To mark the end of a sentence
- **B.** To fill frames with no new output and to separate genuine double letters, which would otherwise merge
- **C.** To represent unknown words
- **D.** To pad batches to the same length

<details>
<summary>Answer</summary>

**B.** To fill frames with no new output and to separate genuine double letters, which would otherwise merge

Without blanks, `ll` would always collapse to one `l`, and every frame would have to emit a letter.

- **A:** CTC has no end-of-sentence role for the blank.
- **B:** Correct.
- **C:** Unknown words are handled by the vocabulary (characters or subwords).
- **D:** Padding is handled by length arguments, not by the blank.

</details>

## 3. Calculation (hard)

Two frames, vocabulary {blank, a}. Frame 1: P(blank) = 0.3, P(a) = 0.7. Frame 2: P(blank) = 0.6, P(a) = 0.4. What is P("a") under CTC? (2 decimals)

<details>
<summary>Answer</summary>

**0.82** (within ±0.005)

Paths collapsing to "a": `aa` $0.7 \times 0.4 = 0.28$, `a_` $0.7 \times 0.6 = 0.42$, `_a` $0.3 \times 0.4 = 0.12$. Sum $0.82$.

</details>

## 4. Multiple choice (medium)

Why can greedy CTC decoding return a transcript that isn't the most probable one?

- **A.** Greedy decoding ignores the blank token
- **B.** It picks the single best path, but a transcript's probability is the sum over many paths, and another transcript can have more total mass
- **C.** Greedy decoding is only an approximation of the softmax
- **D.** It always returns an empty string

<details>
<summary>Answer</summary>

**B.** It picks the single best path, but a transcript's probability is the sum over many paths, and another transcript can have more total mass

Two frames at P(blank) = 0.6 give greedy "" (0.36) while "a" sums to 0.64 over three paths.

- **A:** Greedy decoding uses blanks and then removes them.
- **B:** Correct.
- **C:** It uses the softmax outputs exactly; the issue is path versus transcript.
- **D:** It returns whatever the argmax path collapses to.

</details>

## 5. Calculation (medium)

A 25-word reference is transcribed with 3 substitutions, 2 deletions, and 1 insertion. What is the WER? (2 decimals)

<details>
<summary>Answer</summary>

**0.24** (within ±0.001)

$(3 + 2 + 1) / 25 = 6 / 25 = 0.24$.

</details>

## 6. Multiple choice (medium)

Can word error rate exceed 1.0?

- **A.** No, it is a fraction of reference words
- **B.** Yes, because insertions are not limited by the reference length
- **C.** Only if the reference is empty
- **D.** Only for character error rate

<details>
<summary>Answer</summary>

**B.** Yes, because insertions are not limited by the reference length

A hypothesis that adds many words that weren't said can have more errors than reference words.

- **A:** Insertions are counted in the numerator but not limited by N.
- **B:** Correct.
- **C:** An empty reference makes WER undefined, not merely above 1.
- **D:** Word error rate can too.

</details>

## 7. Multiple choice (medium)

Which architecture is the usual choice for low-latency streaming recognition on a device?

- **A.** Attention encoder-decoder over a 30-second window
- **B.** Streaming transducer (RNN-T style), or chunked CTC
- **C.** Any model, as long as beam width is large
- **D.** A text LLM with no audio encoder

<details>
<summary>Answer</summary>

**B.** Streaming transducer (RNN-T style), or chunked CTC

Transducers and chunked CTC process audio left to right and emit tokens as it arrives.

- **A:** Full-window attention needs the whole chunk before decoding, adding latency.
- **B:** Correct.
- **C:** Beam width doesn't make a non-streaming model stream.
- **D:** A text model can't hear audio.

</details>

## 8. Multiple choice (hard)

A Whisper-based transcriber outputs fluent sentences during long silences. What is the best first fix?

- **A.** Increase the beam size
- **B.** Run voice activity detection and only transcribe speech regions; reject low-confidence or repetitive segments
- **C.** Switch the language to auto-detect
- **D.** Lower the audio volume

<details>
<summary>Answer</summary>

**B.** Run voice activity detection and only transcribe speech regions; reject low-confidence or repetitive segments

Autoregressive decoders can hallucinate on non-speech. Don't feed them silence, and use their confidence signals.

- **A:** A bigger beam still decodes from silence.
- **B:** Correct.
- **C:** Language ID errors on silence often make things worse.
- **D:** Volume doesn't stop the decoder from generating text.

</details>

## 9. Multiple choice (medium)

Your system's WER is 12 percent; a paper reports 4 percent on the same test set. What should you check first?

- **A.** Whether your GPU is slower
- **B.** Whether both used the same text normalization (casing, punctuation, numbers) before computing WER
- **C.** Whether the paper used more beams
- **D.** Whether your audio files are larger

<details>
<summary>Answer</summary>

**B.** Whether both used the same text normalization (casing, punctuation, numbers) before computing WER

Unnormalized punctuation and casing differences count as errors and can inflate WER several-fold.

- **A:** Speed doesn't change WER.
- **B:** Correct.
- **C:** Beam size matters a little; normalization can matter a lot more.
- **D:** File size isn't relevant.

</details>

## 10. Match (medium)

Match each formulation with its defining property.

| Concept | Options |
|---|---|
| CTC | Autoregressive decoder cross-attends to all audio; strong offline, can hallucinate |
| RNN-T (transducer) | Prediction network over past tokens; streams left to right |
| Attention encoder-decoder | Independent per-frame predictions with a blank; fast, non-autoregressive |

<details>
<summary>Answer</summary>

- CTC → Independent per-frame predictions with a blank; fast, non-autoregressive
- RNN-T (transducer) → Prediction network over past tokens; streams left to right
- Attention encoder-decoder → Autoregressive decoder cross-attends to all audio; strong offline, can hallucinate

These are the three standard ways to align audio frames with text.

</details>
