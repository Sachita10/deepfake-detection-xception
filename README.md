# DeepFake Detection using Xception

## Project Overview

This project focuses on detecting deepfake images and videos using the Xception deep learning architecture.

The model was trained on the FaceForensics++ dataset and evaluated on multiple datasets to study generalization performance.

## Dataset

- FaceForensics++ (C23)
- Indian Face Dataset
- HAV-DF Dataset

## Project Workflow

1. Extract frames from FaceForensics++ videos
2. Create balanced real and fake datasets
3. Split data into train, validation and test sets
4. Train Xception model
5. Save model checkpoints
6. Evaluate on FaceForensics++ test set
7. Evaluate on Indian dataset
8. Evaluate on HAV-DF dataset
9. Compare with DeepSafe pretrained model

## Repository Structure

```text
extract_frames.py
prepare_dataset.py
train_xception.py

test_xception_ffpp.py
test_xception_indian.py
test_xception_havdf.py

test_deepsafe.py
```

## Model

- Architecture: Xception
- Classes: Real, Fake
- Input Size: 299 × 299
- Optimizer: Adam
- Learning Rate: 1e-4

## Results

- Best validation accuracy obtained during training.
- Evaluated on multiple datasets for robustness testing.

## Author

Sachita
