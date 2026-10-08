<!-- GENERATED from density-hierarchical-and-mixture-clustering.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: DBSCAN, hierarchical clustering, and Gaussian mixtures

Covers the lesson [DBSCAN, hierarchical clustering, and Gaussian mixtures](../lessons/07-unsupervised/02-density-hierarchical-and-mixture-clustering.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/density-hierarchical-and-mixture-clustering/) grades these interactively and tracks a review queue.

## 1. Multiple choice (medium)

DBSCAN with ε = 1 and min_samples = 3. Point P has two other points within distance 1. What is P?

- **A.** Noise
- **B.** A core point
- **C.** A border point
- **D.** It depends on K

<details>
<summary>Answer</summary>

**B.** A core point

Counting P itself, there are 3 points within ε, which meets min_samples, so P is core.

</details>

## 2. Calculation (medium)

Points 0, 0.5, 1.0, 5.0, 9.0, 9.4 with ε = 0.6 and min_samples = 3. How many points are noise?

<details>
<summary>Answer</summary>

**3**

Only 0.5 is core (0, 0.5, 1.0); 0 and 1.0 are its border points. 5.0 is isolated; 9.0 and 9.4 have only 2 points each in range and no core neighbor. Noise = 5.0, 9.0, 9.4.

</details>

## 3. Match (medium)

Match each need to a method.

| Concept | Options |
|---|---|
| Crescent-shaped clusters plus outliers | K-means |
| Probability that a point belongs to each cluster | Hierarchical clustering |
| Explore nested groupings at several levels | Gaussian mixture |
| Millions of points, roughly round clusters | DBSCAN |

<details>
<summary>Answer</summary>

- Crescent-shaped clusters plus outliers → DBSCAN
- Probability that a point belongs to each cluster → Gaussian mixture
- Explore nested groupings at several levels → Hierarchical clustering
- Millions of points, roughly round clusters → K-means

Pick by cluster shape, need for soft assignments, nesting, and scale.

</details>

## 4. Calculation (hard)

A two-component mixture has weights 0.5 and 0.5. At point $x$, the component densities are 0.2 and 0.05. What is the responsibility of component 1?

<details>
<summary>Answer</summary>

**0.8** (within ±0.0001)

$\frac{0.5 \times 0.2}{0.5 \times 0.2 + 0.5 \times 0.05} = \frac{0.1}{0.125} = 0.8$.

</details>

## 5. Multiple choice (medium)

A dataset has one very dense cluster and one sparse cluster. Why might DBSCAN struggle?

- **A.** DBSCAN requires K.
- **B.** A single ε small enough for the dense cluster labels the sparse one as noise; one large enough for the sparse cluster merges things in the dense region.
- **C.** DBSCAN only works in 2D.
- **D.** It can't handle more than two clusters.

<details>
<summary>Answer</summary>

**B.** A single ε small enough for the dense cluster labels the sparse one as noise; one large enough for the sparse cluster merges things in the dense region.

HDBSCAN handles varying densities better.

</details>

## 6. Reflection (medium)

How could you use a Gaussian mixture model for anomaly detection, and what still needs a human decision?

<details>
<summary>Answer</summary>

**Model answer.** Fit the GMM on normal data, then score new points by their likelihood (or negative log-likelihood); unusually low likelihood means the point doesn't fit any learned component. The threshold that turns a score into an alert still has to be chosen, ideally with labeled past incidents and the cost of false alarms versus misses.

Scores rank unusualness; thresholds are product decisions.

</details>
