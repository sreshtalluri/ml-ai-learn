---
title: Computer vision and multimodal
summary: "Detect, segment, and encode images; learn visual features without labels; align images with text; and connect vision encoders to language models."
skill: deep-learning
---

# Module 25: Computer vision and multimodal

Convolutional networks taught us to classify a whole image. This module covers what production vision systems actually do: find and outline objects, encode images with transformers, learn from unlabelled pictures, put images and text in one shared space, and let a language model read images.

| Task | Output | Typical model | Metric |
|---|---|---|---|
| Detection | labelled boxes | YOLO family, DETR family | mAP@0.5:0.95 |
| Semantic segmentation | class per pixel | U-Net, SegFormer | per-class IoU, Dice |
| Representation learning | feature vector per image | ViT with SimCLR, DINO, MAE | linear probe, k-NN |
| Image-text matching | shared embedding space | CLIP, SigLIP | zero-shot accuracy, recall@k |
| Visual question answering | text | ViT + projector + LLM | grounded accuracy, hallucination rate |

| # | Lesson | You will be able to |
|---|---|---|
| 1 | [Object detection and YOLO](01-object-detection-yolo.md) | decode a YOLO head, compute IoU, run NMS by hand, and read mAP |
| 2 | [Semantic segmentation and U-Net](02-semantic-segmentation-unet.md) | trace U-Net shapes, explain skip connections, combine cross-entropy with Dice, and report per-class IoU |
| 3 | [Vision transformers](03-vision-transformers.md) | turn an image into patch tokens, count ViT parameters, and budget attention cost by patch size |
| 4 | [Self-supervised vision](04-self-supervised-vision.md) | compute InfoNCE, explain collapse and how DINO and MAE avoid it, and evaluate with linear probes |
| 5 | [CLIP and contrastive image-text pretraining](05-clip-contrastive-pretraining.md) | compute the similarity matrix, both softmaxes, and the symmetric loss; run zero-shot classification; compare CLIP with SigLIP |
| 6 | [Vision-language models](06-vision-language-models.md) | trace image tokens into an LLM, budget tokens and KV-cache memory, and evaluate visual hallucination |

**Interactive lab:** [Contrastive image-text matching](https://sreshtalluri.github.io/ml-ai-learn/labs/clip-similarity/)
