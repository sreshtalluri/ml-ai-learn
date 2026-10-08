<!-- GENERATED from k-means.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: K-means clustering

Covers the lesson [K-means clustering](../lessons/07-unsupervised/01-k-means.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/k-means/) grades these interactively and tracks a review queue.

## 1. Fill in (easy)

Cluster 1 contains the points $(2, 4)$, $(4, 6)$, and $(6, 2)$. After the update step, what is its centroid? Answer as `(x, y)`.

<details>
<summary>Answer</summary>

**(4, 4)** or **(4,4)** or **4,4** or **4, 4**

The mean of the x-coordinates is $(2+4+6)/3 = 4$ and of the y-coordinates $(4+6+2)/3 = 4$, so $\mu_1 = (4, 4)$.

</details>

## 2. Calculation (medium)

Using that centroid $(4, 4)$, what is the cluster's contribution to inertia (sum of squared distances)?

<details>
<summary>Answer</summary>

**16** (within ±0.001)

$(2,4)$: $4 + 0 = 4$. $(4,6)$: $0 + 4 = 4$. $(6,2)$: $4 + 4 = 8$. Total $= 16$.

</details>

## 3. Multiple choice (easy)

Centroids are $\mu_1 = (0, 0)$ and $\mu_2 = (6, 0)$. Which cluster does the point $(2, 3)$ join?

- **A.** Cluster 1, squared distance 13 versus 25
- **B.** Cluster 2, squared distance 13 versus 25
- **C.** Cluster 1, because it was created first
- **D.** It is a tie

<details>
<summary>Answer</summary>

**A.** Cluster 1, squared distance 13 versus 25

To $\mu_1$: $4 + 9 = 13$. To $\mu_2$: $16 + 9 = 25$. The nearer centroid wins.

</details>

## 4. Arrange in order (easy)

Put the K-means algorithm in order.

- Repeat assignment and update until assignments stop changing
- Move every centroid to the mean of its assigned points
- Assign every point to its nearest centroid
- Choose K and initialize K centroids (for example with k-means++)

<details>
<summary>Answer</summary>

1. Choose K and initialize K centroids (for example with k-means++)
2. Assign every point to its nearest centroid
3. Move every centroid to the mean of its assigned points
4. Repeat assignment and update until assignments stop changing

Initialize, then alternate the two steps until convergence.

</details>

## 5. Multiple choice (medium)

A teammate picks K = 40 for 50,000 customers because "inertia was lowest at K = 40, the largest value I tried." What is wrong with this reasoning?

- **A.** Nothing; lower inertia always means better clusters.
- **B.** Inertia always decreases as K grows, so it cannot choose K by itself.
- **C.** Inertia increases with K, so K = 40 must be a mistake.
- **D.** K-means cannot run with more than 10 clusters.

<details>
<summary>Answer</summary>

**B.** Inertia always decreases as K grows, so it cannot choose K by itself.

With K = n every point is its own cluster and inertia is 0. Use the elbow, silhouette, stability across seeds, and whether the clusters are actionable.

- **A:** Lower inertia is guaranteed by more clusters, regardless of real structure.
- **B:** Correct.
- **C:** The opposite is true.
- **D:** There is no such limit.

</details>

## 6. Select all that apply (hard)

For which datasets would K-means likely produce misleading clusters? Select all that apply.

- **A.** Two concentric rings
- **B.** Three round, well-separated blobs of similar size
- **C.** One huge diffuse group next to one tiny dense group
- **D.** Two long, parallel, elongated bands

<details>
<summary>Answer</summary>

**A, C, D**

K-means draws straight boundaries halfway between centroids and prefers round clusters of similar size. Rings, very unequal sizes, and elongated bands all violate those assumptions. Round, separated blobs are its ideal case.

- **A:** Correct. No pair of centroids can separate an inner ring from an outer one.
- **B:** This is exactly the case K-means is designed for.
- **C:** Correct. It tends to split the big group and absorb the small one.
- **D:** Correct. It often cuts across the bands instead of along them.

</details>

## 7. Multiple choice (medium)

You run K-means twice on the same data with different random seeds and get inertias of 462.8 and 2359.7. What is the best explanation and fix?

- **A.** The data changed between runs; reload it.
- **B.** K-means converged to different local optima; use k-means++ and several initializations, keeping the best.
- **C.** The second run has a bug; K-means is deterministic.
- **D.** Inertia is random noise; average the two values.

<details>
<summary>Answer</summary>

**B.** K-means converged to different local optima; use k-means++ and several initializations, keeping the best.

Lloyd's algorithm only guarantees a local optimum. Initialization decides which one. `n_init=10` with k-means++ is the standard defense.

- **A:** Same data was used; the difference comes from initialization.
- **B:** Correct.
- **C:** K-means depends on its random initialization unless the seed is fixed.
- **D:** Inertia is a deterministic function of the final clustering, not noise.

</details>

## 8. Reflection (hard)

Marketing wants to launch a campaign for "cluster 3: bargain hunters" found by K-means. What would you check before agreeing?

<details>
<summary>Answer</summary>

**Model answer.** Whether the cluster is real and stable: rerun with different seeds and subsamples and see if it reappears; check silhouette and whether a different K changes it. Whether features were scaled and chosen sensibly. Whether the "bargain hunter" label is supported by the features that distinguish the cluster (look at centroid values), and validate with domain experts or a small experiment. A K-means cluster is a hypothesis, not a verified customer type, and cluster numbers change between runs.

Distinguish a discovered cluster from a verified real-world category.

</details>
