# Adapted from AI Engineering from Scratch (MIT), phases/04-computer-vision/06-object-detection-yolo
"""Object detection pieces in NumPy: IoU, non-maximum suppression, YOLO-style box encode/decode,
target assignment, and average precision at IoU 0.5.

All boxes and scores are SYNTHETIC, chosen by hand so the arithmetic is checkable. Deterministic.
Run from guide/:  uv run code/25-computer-vision/detection_scratch.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, TEAL, plt, save  # noqa: E402


# ---------- 1. IoU: the similarity measure for boxes ----------
def box_iou(a, b):
    """a: [Na, 4], b: [Nb, 4] boxes as (x1, y1, x2, y2). Returns [Na, Nb] pairwise IoU."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])    # intersection corners, broadcast to [Na, Nb]
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)   # 0 when boxes don't overlap
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


# ---------- 2. NMS: keep the best box, drop its near-duplicates, repeat ----------
def nms(boxes, scores, iou_threshold=0.5):
    order = np.argsort(-scores)                        # highest score first
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        ious = box_iou(boxes[[i]], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]      # survivors: boxes that overlap i too little
    return keep


# ---------- 3. YOLO-style encode/decode relative to a grid cell and an anchor ----------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


def encode(box, cell_x, cell_y, stride, anchor_wh):
    """Pixel box (x1, y1, x2, y2) -> regression targets (tx, ty, tw, th)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = box[2] - box[0], box[3] - box[1]
    off_x, off_y = cx / stride - cell_x, cy / stride - cell_y      # centre offset inside the cell, in [0, 1)
    return np.array([logit(off_x), logit(off_y), np.log(w / anchor_wh[0]), np.log(h / anchor_wh[1])])


def decode(t, cell_x, cell_y, stride, anchor_wh):
    cx = (sigmoid(t[0]) + cell_x) * stride            # sigmoid keeps the centre inside its cell
    cy = (sigmoid(t[1]) + cell_y) * stride
    w = anchor_wh[0] * np.exp(t[2])                   # exp keeps width positive, scales the anchor
    h = anchor_wh[1] * np.exp(t[3])
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def assign(box, anchors, stride):
    """Which grid cell and which anchor are responsible for this ground-truth box?"""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = box[2] - box[0], box[3] - box[1]
    shape_iou = [min(w, aw) * min(h, ah) / (w * h + aw * ah - min(w, aw) * min(h, ah)) for aw, ah in anchors]
    return int(cx // stride), int(cy // stride), int(np.argmax(shape_iou)), shape_iou


# ---------- 4. Average precision at one IoU threshold ----------
def average_precision(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5):
    order = np.argsort(-pred_scores)
    matched = set()
    tp = []
    for i in order:                                    # greedy: each GT box can be matched once
        ious = box_iou(pred_boxes[[i]], gt_boxes)[0]
        j = int(np.argmax(ious))
        hit = ious[j] >= iou_thr and j not in matched
        if hit:
            matched.add(j)
        tp.append(hit)
    tp = np.array(tp, dtype=float)
    precision = np.cumsum(tp) / np.arange(1, len(tp) + 1)
    recall = np.cumsum(tp) / len(gt_boxes)
    # area under the precision envelope (all-point interpolation, as in COCO/VOC2010+)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_steps = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(envelope * recall_steps)), precision, recall, tp


if __name__ == "__main__":
    print("== IoU ==")
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[5, 0, 15, 10]], dtype=float)
    print(f"half-overlap boxes: IoU = {box_iou(a, b)[0, 0]:.4f}  (50 / (100 + 100 - 50) = 1/3)")
    p = np.array([[50, 50, 150, 150]], dtype=float)
    g = np.array([[60, 60, 160, 160]], dtype=float)
    print(f"worked example: IoU = {box_iou(p, g)[0, 0]:.4f}  (8100 / 11900)")

    print("\n== NMS ==")
    boxes = np.array([[50, 50, 150, 150], [60, 60, 160, 160], [55, 45, 150, 145],
                      [200, 80, 280, 200], [210, 90, 285, 205]], dtype=float)
    scores = np.array([0.92, 0.85, 0.60, 0.80, 0.55])
    print("pairwise IoU:\n", box_iou(boxes, boxes).round(3))
    keep = nms(boxes, scores, 0.5)
    print(f"kept (threshold 0.5): {keep}")

    print("\n== encode / decode ==")
    anchors = [(30, 60), (75, 170), (200, 380)]
    stride = 32
    gt = np.array([120, 80, 240, 220], dtype=float)
    gx, gy, k, shape_iou = assign(gt, anchors, stride)
    print(f"gt box {gt.tolist()} -> centre (180, 150) -> cell ({gx}, {gy}); "
          f"shape IoU with anchors {np.round(shape_iou, 3).tolist()} -> anchor {k} {anchors[k]}")
    t = encode(gt, gx, gy, stride, anchors[k])
    back = decode(t, gx, gy, stride, anchors[k])
    print(f"targets (tx, ty, tw, th) = {t.round(4).tolist()}")
    print(f"decoded back: {back.round(4).tolist()}")
    print(f"output tensor for a 416x416 input, stride 32, 3 anchors, 80 classes: "
          f"[{416 // stride}, {416 // stride}, 3, {5 + 80}] = {(416 // stride) ** 2 * 3 * 85:,} numbers")

    print("\n== AP@0.5 ==")
    gts = np.array([[10, 10, 50, 50], [100, 100, 150, 160], [200, 20, 260, 80]], dtype=float)
    preds = np.array([[12, 12, 52, 50],      # good hit on gt 0
                      [101, 98, 149, 158],   # good hit on gt 1
                      [14, 8, 50, 52],       # duplicate of gt 0 (NMS missed it) -> false positive
                      [300, 300, 340, 340],  # nothing there -> false positive
                      [205, 25, 262, 85]],   # hit on gt 2, low score
                     dtype=float)
    pscores = np.array([0.95, 0.90, 0.70, 0.60, 0.40])
    ap, prec, rec, tp = average_precision(preds, pscores, gts)
    print("rank  score  TP  precision  recall")
    for r in range(len(tp)):
        print(f"{r + 1:>4}  {np.sort(pscores)[::-1][r]:.2f}   {int(tp[r])}   {prec[r]:.3f}      {rec[r]:.3f}")
    print(f"AP@0.5 = {ap:.4f}")

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, title, idx in [(axes[0], "Before NMS: 5 raw boxes", range(5)), (axes[1], "After NMS (IoU > 0.5 removed)", keep)]:
        ax.set(xlim=(30, 300), ylim=(230, 30), title=title, aspect="equal")
        ax.grid(False)
        for i in idx:
            x1, y1, x2, y2 = boxes[i]
            color = BLUE if i < 3 else ORANGE
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=2 if i in keep else 1.2,
                                       ls="-" if i in keep else "--", edgecolor=color))
            ax.text(x1 + 2, y1 - 3, f"{scores[i]:.2f}", fontsize=8, color=color)
    axes[0].text(35, 225, "SYNTHETIC boxes: blue = object A, orange = object B", fontsize=8, color=GRAY)
    axes[1].text(35, 225, f"kept boxes {keep}: one per object", fontsize=8, color=TEAL)
    save(fig, "object-detection-yolo")
