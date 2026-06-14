# Baseline Report – DeepFake Detection

## Objective

Evaluate the baseline performance of existing deepfake detection models before further improvements.

---

## Model 1: Xception

### Dataset
- FaceForensics++ C23

### Accuracy

| Dataset | Accuracy |
|----------|----------|
| FaceForensics++ Test | 74.22% |
| Indian Dataset | 73% |
| HAV-DF | 43% |

### Speed
| Task | Approximate Time |
|--------|--------|
| Single Image Prediction | 10–30 ms/image |
| Batch Inference | Real-time capable |
| HAV-DF Video Prediction | 2–5 sec/video |


### Failure Cases

Observed errors:

- Misclassified low-quality fake videos as real.
- Performance dropped on unseen Indian faces.
- Struggled with heavily compressed videos.
- Incorrect predictions when faces occupied very small regions.

---

## Model 2: DeepSafe

### Accuracy

| Dataset | Accuracy |
|----------|----------|
| Indian Dataset | 86% |
| HAV-DF | 34% |

### Speed

| Task | Approximate Time |
|--------|--------|
| Single Image Prediction | 15–40 ms/image |
| Video Prediction | 3–6 sec/video |


### Failure Cases

Observed errors:

- False positives on highly edited real images.
- Reduced robustness to lighting variations.
- Some fake videos detected as real.

---



---

## Conclusion

The baseline evaluation shows that both models perform reasonably well on FaceForensics++ but experience performance degradation on unseen datasets such as Indian faces and HAV-DF. Future work will focus on improving cross-dataset generalization and reducing false positives.
Note: Speed values are approximate observations from local testing on Apple Silicon hardware and are intended for baseline comparison only.
