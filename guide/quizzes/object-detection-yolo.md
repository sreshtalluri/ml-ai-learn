<!-- GENERATED from object-detection-yolo.yml by web/scripts/gen-quizzes.mjs. Edit the .yml, then run: npm run gen:quizzes -->
# Quiz: Object detection and YOLO

Covers the lesson [Object detection and YOLO](../lessons/25-computer-vision/01-object-detection-yolo.md). Try each question before opening the answer. The [website](https://sreshtalluri.github.io/ml-ai-learn/quizzes/object-detection-yolo/) grades these interactively and tracks a review queue.

## 1. Multiple choice (easy)

A YOLO-style head uses a 13 by 13 grid, 3 anchors per cell, and 20 classes. Its output is shaped `[13, 13, 3, 25]`. Why is the last dimension 25?

- **A.** It is padded to a convenient size
- **B.** 25 is the number of classes in PASCAL VOC
- **C.** 4 box coordinates + 1 bias + 20 classes
- **D.** 4 box offsets + 1 objectness score + 20 class scores

<details>
<summary>Answer</summary>

**D.** 4 box offsets + 1 objectness score + 20 class scores

Every anchor predicts $(t_x, t_y, t_w, t_h)$, one objectness logit, and one score per class, so the per-anchor size is $5 + C = 5 + 20 = 25$. With COCO's 80 classes it would be 85.

- **A:** Nothing is padded; every number has a meaning.
- **B:** VOC has 20 classes, not 25.
- **C:** The fifth number is objectness ("is an object centred here?"), not a bias term.
- **D:** Correct.

</details>

## 2. Multiple choice (medium)

Why does YOLO pass $t_x$ and $t_y$ through a sigmoid before adding the cell index?

- **A.** Convolutions require a sigmoid on every output
- **B.** To prevent NaNs during training
- **C.** To keep the predicted centre inside the cell that owns the object
- **D.** To make the box larger

<details>
<summary>Answer</summary>

**C.** To keep the predicted centre inside the cell that owns the object

$\sigma(t_x) \in (0, 1)$, so $(\sigma(t_x) + c_x)\,s$ always lies inside cell $c_x$. Each cell then only predicts objects centred in its own area, which keeps assignment and training stable.

- **A:** Convolutions have no such requirement; the sigmoid is a design choice for this output.
- **B:** NaNs are not the issue; an unbounded offset would let cells predict objects anywhere.
- **C:** Correct.
- **D:** Box size comes from the exponential on $t_w, t_h$, not the sigmoid.

</details>

## 3. Calculation (medium)

Prediction $(0, 0, 40, 40)$ and ground truth $(10, 10, 50, 50)$, both as $(x_1, y_1, x_2, y_2)$. Compute the IoU to three decimals.

<details>
<summary>Answer</summary>

**0.391** (within ±0.002)

Intersection is $30 \times 30 = 900$. Each box is $1600$. Union $= 1600 + 1600 - 900 = 2300$. IoU $= 900/2300 = 0.391$, below the usual 0.5 threshold.

</details>

## 4. Multiple choice (easy)

The head outputs 15 overlapping boxes around one object. After NMS with threshold 0.45 you get one box. What did NMS do?

- **A.** Averaged the 15 boxes into one
- **B.** Kept the highest-scoring box and deleted every box overlapping it with IoU above 0.45
- **C.** Picked one box at random
- **D.** Merged boxes whose classes matched

<details>
<summary>Answer</summary>

**B.** Kept the highest-scoring box and deleted every box overlapping it with IoU above 0.45

NMS is greedy deletion. Sort by score, keep the top box, remove everything that overlaps it above the threshold, repeat on what remains. It never averages or merges.

- **A:** Averaging is "weighted box fusion", a different technique.
- **B:** The choice is deterministic, by score.
- **C:** Correct.
- **D:** NMS does not merge. Class matters only in deciding which boxes may suppress each other (per-class NMS).

</details>

## 5. Multiple choice (medium)

A detector scores mAP@0.5 = 0.75 but mAP@0.5:0.95 = 0.32. What is the most accurate reading?

- **A.** The dataset has too many classes
- **B.** The model finds the right objects but its boxes are loose, so it fails the stricter IoU thresholds
- **C.** The model is broken; the two numbers should be equal
- **D.** The confidence threshold is too high

<details>
<summary>Answer</summary>

**B.** The model finds the right objects but its boxes are loose, so it fails the stricter IoU thresholds

mAP@0.5 is lenient about localization. mAP@0.5:0.95 averages over thresholds up to 0.95. A large gap means objects are found but boxes are not tight. Fixes target localization, such as an IoU-based box loss or higher-resolution features.

- **A:** Class count affects both numbers similarly.
- **B:** The stricter metric is always lower or equal; a gap is normal, its size is the signal.
- **C:** Correct.
- **D:** Evaluation uses a very low threshold and the full ranked list; this would not create a gap between IoU thresholds.

</details>

## 6. Multiple choice (medium)

YOLOv1 used $\lambda_{\text{coord}} = 5$ and $\lambda_{\text{noobj}} = 0.5$. What do these weights compensate for?

- **A.** Most (cell, anchor) pairs are empty, so their objectness loss must be down-weighted or it dominates; the box loss is up-weighted so localization still gets gradient
- **B.** GPU memory limits
- **C.** The number of anchors per cell
- **D.** They were arbitrary

<details>
<summary>Answer</summary>

**A.** Most (cell, anchor) pairs are empty, so their objectness loss must be down-weighted or it dominates; the box loss is up-weighted so localization still gets gradient

A 13 by 13 grid with 3 anchors has 507 pairs, and usually only a handful own objects. Without a small $\lambda_{\text{noobj}}$ the "stay silent" loss swamps everything else.

- **A:** Correct.
- **B:** Loss weights do not change memory.
- **C:** Anchor count changes how many empty pairs exist, but the weights address the imbalance, not the count itself.
- **D:** They encode a real imbalance, explained in the paper.

</details>

## 7. Calculation (hard)

Two ground-truth objects. Three detections sorted by score: rank 1 is a true positive, rank 2 a false positive, rank 3 a true positive. Using the precision envelope (best precision at that recall or higher), what is AP?

<details>
<summary>Answer</summary>

**0.8333** (within ±0.002)

Precision: $1/1, 1/2, 2/3$. Recall: $0.5, 0.5, 1.0$. Envelope: rank 1 keeps 1.0; rank 3 is $0.667$. Recall rises by 0.5 at rank 1 and 0.5 at rank 3, so AP $= 1.0 \times 0.5 + 0.667 \times 0.5 = 0.833$.

</details>

## 8. Multiple choice (hard)

Recall is high for isolated people but low in dense crowds. Which change is the most direct first experiment?

- **A.** Increase the learning rate
- **B.** Raise the NMS IoU threshold for the person class (or try Soft-NMS) and re-measure crowd recall
- **C.** Lower the confidence threshold for display
- **D.** Add more classes

<details>
<summary>Answer</summary>

**B.** Raise the NMS IoU threshold for the person class (or try Soft-NMS) and re-measure crowd recall

Real people in a crowd can overlap with IoU above 0.5, so NMS deletes true detections as "duplicates". Testing a higher threshold or Soft-NMS directly checks that hypothesis. Assignment collisions and resolution are the next suspects.

- **A:** Nothing points to an optimization problem.
- **B:** Correct.
- **C:** The boxes are being deleted by NMS, not hidden by the display threshold.
- **D:** Unrelated to crowd recall.

</details>

## 9. Multiple choice (medium)

You must detect 15 product types on a shelf camera at 30 frames per second on a small edge device. Which plan fits best?

- **A.** A large two-stage detector at full resolution
- **B.** A small one-stage detector fine-tuned on labelled shelf images, exported to an optimized runtime, with latency measured including decode and NMS
- **C.** A vision-language model prompted per frame
- **D.** Zero-shot CLIP on the whole frame

<details>
<summary>Answer</summary>

**B.** A small one-stage detector fine-tuned on labelled shelf images, exported to an optimized runtime, with latency measured including decode and NMS

One-stage detectors are built for real-time boxes. Fine-tuning on your own images handles the domain, and end-to-end latency must include post-processing.

- **A:** Too slow for 30 frames per second on an edge device.
- **B:** Correct.
- **C:** VLMs are far too slow per frame and poor at precise boxes and counting.
- **D:** CLIP scores whole images or crops; it does not output boxes.

</details>
