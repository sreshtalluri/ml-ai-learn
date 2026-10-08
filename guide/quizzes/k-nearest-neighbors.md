<!-- GENERATED from k-nearest-neighbors.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: K-nearest neighbors

Covers the lesson [K-nearest neighbors](../lessons/05-instance-and-probabilistic/01-k-nearest-neighbors.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/k-nearest-neighbors/) grades these interactively and tracks a review queue.

## 1. Calculation (easy)

What is the Euclidean distance between $q = (1, 2)$ and $x = (4, 6)$?

<details>
<summary>Answer</summary>

**5** (within ±0.001)

Differences are $3$ and $4$. $\sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$.

</details>

## 2. Calculation (easy)

What is the Manhattan distance between $q = (1, 2, 0)$ and $x = (4, 6, -2)$?

<details>
<summary>Answer</summary>

**9** (within ±0.001)

$|1-4| + |2-6| + |0-(-2)| = 3 + 4 + 2 = 9$.

</details>

## 3. Multiple choice (medium)

The five nearest neighbors of a query, from nearest to farthest, have labels `[1, 0, 0, 1, 0]`. What do K = 1, K = 3, and K = 5 predict?

- **A.** 1, 0, 0
- **B.** 1, 1, 0
- **C.** 0, 0, 0
- **D.** 1, 0, 1

<details>
<summary>Answer</summary>

**A.** 1, 0, 0

K = 1 uses only the nearest label, 1. K = 3 uses `[1, 0, 0]`, majority 0. K = 5 uses `[1, 0, 0, 1, 0]`: three 0s and two 1s, majority 0.

</details>

## 4. Multiple choice (medium)

Features are `age` (20 to 70) and `annual_income` in dollars (20,000 to 200,000). You run KNN without scaling. What happens?

- **A.** Nothing; KNN is scale-invariant.
- **B.** Distances are dominated by income, so the model effectively ignores age.
- **C.** Distances are dominated by age because it has fewer distinct values.
- **D.** The model raises an error about mismatched units.

<details>
<summary>Answer</summary>

**B.** Distances are dominated by income, so the model effectively ignores age.

An income gap of 10,000 contributes $10{,}000^2$ to the squared distance, while the largest possible age gap contributes $50^2 = 2{,}500$. Income decides who the neighbors are. Standardize features first.

- **A:** Distance-based models are very sensitive to scale; tree models are the scale-invariant ones.
- **B:** Correct.
- **C:** The number of distinct values does not matter; the magnitude of the differences does.
- **D:** Nothing checks units. The failure is silent, which is what makes it dangerous.

</details>

## 5. Match (medium)

Match each setting to its typical behavior.

| Concept | Options |
|---|---|
| K = 1 | Closer neighbors have more influence |
| K = n (all points) | Balances bias and variance on held-out folds |
| K chosen by cross-validation | Always predicts the majority class |
| Distance-weighted votes | Zero training error, high variance, jagged boundary |

<details>
<summary>Answer</summary>

- K = 1 → Zero training error, high variance, jagged boundary
- K = n (all points) → Always predicts the majority class
- K chosen by cross-validation → Balances bias and variance on held-out folds
- Distance-weighted votes → Closer neighbors have more influence

K = 1 memorizes; K = n ignores the query entirely (every query has the same neighbors). Cross-validation picks K on data the model did not see, and distance weighting softens the cutoff at the K-th neighbor.

</details>

## 6. Select all that apply (hard)

A KNN fraud model looked great in a notebook but in production it is slow (300 ms per request) and its precision dropped. Training data is 4 million transactions with 120 raw features. Which changes address real causes? Select all that apply.

- **A.** Use an approximate nearest-neighbor index instead of brute-force search.
- **B.** Reduce or select features (or learn an embedding) to fight the curse of dimensionality.
- **C.** Check that production features are scaled with the training-set mean and standard deviation.
- **D.** Increase K to 4 million to use all the data.

<details>
<summary>Answer</summary>

**A, B, C**

Brute force is $O(np)$ per query: 4M × 120 operations. ANN indexes cut that dramatically. With 120 raw features, distances concentrate and become less informative. A classic production bug is training-serving skew: the notebook scaled features but the service sends raw values.

- **A:** Correct. ANN search (HNSW, IVF) is the standard fix for latency.
- **B:** Correct. Fewer, more relevant dimensions make distances meaningful again.
- **C:** Correct. Unscaled production inputs would silently change every neighbor set.
- **D:** K = n predicts the majority class for every request, which destroys recall on fraud.

</details>

## 7. Multiple choice (medium)

Which pipeline avoids leakage when choosing K with 5-fold cross-validation?

- **A.** Standardize all data, then run cross-validation on KNN.
- **B.** Put StandardScaler and KNN in one pipeline and cross-validate the pipeline.
- **C.** Standardize the test set using its own mean and standard deviation.
- **D.** Skip scaling during cross-validation and add it only in production.

<details>
<summary>Answer</summary>

**B.** Put StandardScaler and KNN in one pipeline and cross-validate the pipeline.

Inside a pipeline, each fold fits the scaler on its training portion only, so validation folds never influence the statistics used to transform them.

- **A:** The scaler saw the validation folds, a mild but real leak.
- **B:** Correct.
- **C:** Test data must use training statistics; using its own is both a leak and a mismatch with production.
- **D:** Then cross-validation measures a different model from the one you deploy.

</details>

## 8. Reflection (hard)

Explain in your own words why KNN tends to get worse as you add many features, even if each new feature is only slightly noisy.

<details>
<summary>Answer</summary>

**Model answer.** Every feature adds a term to every distance. Irrelevant or noisy features add random amounts that swamp the few informative ones. In high dimensions, distances between random points also concentrate: the nearest and the farthest neighbor end up almost equally far, so "nearest" carries little information. The fix is fewer, better features or a learned lower-dimensional representation.

This is the curse of dimensionality, and it affects every distance-based method, including K-means.

</details>
