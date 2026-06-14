import os

dataset_path = os.path.expanduser("~/Downloads/HAV-DF")

video_files = []

for root, dirs, files in os.walk(dataset_path):
    for f in files:
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_files.append(os.path.join(root, f))

print("Total videos found:", len(video_files))
train_path = os.path.expanduser("~/Downloads/HAV-DF/train_videos")
test_path = os.path.expanduser("~/Downloads/HAV-DF/test_videos")

def count_videos(path):
    return len([
        f for f in os.listdir(path)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ])

print("Train videos:", count_videos(train_path))
print("Test videos:", count_videos(test_path))
import os
import cv2
import torch
import timm
import pandas as pd
import numpy as np

from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------
# DEVICE (MPS / CPU)
# ----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = timm.create_model(
    "xception",
    pretrained=False,
    num_classes=2
)

model.load_state_dict(
    torch.load("best_xception.pth", map_location=device)
)

model = model.to(device)
model.eval()

print("Model loaded successfully!")

# ----------------------------
# TRANSFORM (must match training)
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------
# PATHS
# ----------------------------
base_path = os.path.expanduser("~/Downloads/HAV-DF")
video_dir = os.path.join(base_path, "test_videos")
csv_path = os.path.join(base_path, "video_metadata.csv")

# ----------------------------
# LOAD CSV
# ----------------------------
df = pd.read_csv(csv_path)

print("CSV columns:", df.columns)

# ----------------------------
# KEEP ONLY TEST VIDEOS
# ----------------------------
test_videos = set(os.listdir(video_dir))
df = df[df["video_name"].isin(test_videos)].reset_index(drop=True)

print("Total test videos:", len(df))

# ----------------------------
# LABEL MAPPING (FIXED)
# ----------------------------
print("Unique labels:", df["label"].unique())

label_map = {
    "FAKE": 0,
    "REAL": 1
}

df["label_num"] = df["label"].str.upper().map(label_map)

# ----------------------------
# VIDEO PREDICTION FUNCTION
# ----------------------------
def predict_video(video_path, frame_interval=50):

    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    frame_preds = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = transform(frame)
            frame = frame.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(frame)
                pred = torch.argmax(output, dim=1).item()

            frame_preds.append(pred)

        frame_id += 1

    cap.release()

    if len(frame_preds) == 0:
        return None

    # Majority voting
    return int(np.bincount(frame_preds).argmax())

# ----------------------------
# TEST LOOP
# ----------------------------
y_true = []
y_pred = []

for _, row in df.iterrows():

    video_name = row["video_name"]   # FIXED HERE
    true_label = row["label_num"]

    video_path = os.path.join(video_dir, video_name)

    pred_label = predict_video(video_path, frame_interval=50)

    if pred_label is not None:
        y_true.append(true_label)
        y_pred.append(pred_label)

    print(f"{video_name} → True: {true_label}, Pred: {pred_label}")

# ----------------------------
# FINAL METRICS
# ----------------------------
accuracy = accuracy_score(y_true, y_pred)

print("\n====================")
print("FINAL RESULTS")
print("====================")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["fake", "real"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
import pandas as pd

df = pd.read_csv("/Users/sachita/Downloads/HAV-DF/video_metadata.csv")
print(df.columns)
print(df.head())