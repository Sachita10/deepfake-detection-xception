import os
import cv2
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# CONFIG
# =====================================

API_URL = "http://localhost:8000/detect"

base_path = os.path.expanduser("~/Downloads/HAV-DF")
video_dir = os.path.join(base_path, "test_videos")
csv_path = os.path.join(base_path, "video_metadata.csv")

# =====================================
# LOAD METADATA
# =====================================

df = pd.read_csv(csv_path)

print("CSV columns:", df.columns)

# keep only test videos
test_videos = set(os.listdir(video_dir))
df = df[df["video_name"].isin(test_videos)].reset_index(drop=True)

# label mapping (FIXED for HAV-DF)
df["label_num"] = df["label"].str.upper().map({
    "FAKE": 1,
    "REAL": 0
})

print("Total test videos:", len(df))

# =====================================
# STORAGE
# =====================================

y_true = []
y_pred = []
results = []

# =====================================
# API CALL FUNCTION
# =====================================

def predict_frame(image_path):

    with open(image_path, "rb") as f:

        files = {"file": f}

        data = {
            "threshold": 0.5,
            "ensemble_method": "voting",
            "models": "npr_deepfakedetection"
        }

        response = requests.post(
            API_URL,
            files=files,
            data=data,
            timeout=60
        )

    return response.json()

# =====================================
# VIDEO PREDICTION FUNCTION
# =====================================

def predict_video(video_path, frame_interval=50):

    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    frame_preds = []

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:

            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)

            try:
                result = predict_frame(temp_path)

                pred = 1 if result["is_likely_deepfake"] else 0
                frame_preds.append(pred)

            except Exception as e:
                print("Frame error:", e)

        frame_id += 1

    cap.release()

    if len(frame_preds) == 0:
        return None

    # majority voting
    return int(np.bincount(frame_preds).argmax())

# =====================================
# TEST LOOP
# =====================================

for _, row in df.iterrows():

    video_name = row["video_name"]
    true_label = row["label_num"]

    video_path = os.path.join(video_dir, video_name)

    pred_label = predict_video(video_path, frame_interval=50)

    if pred_label is not None:
        y_true.append(true_label)
        y_pred.append(pred_label)

    print(f"{video_name} → True: {true_label}, Pred: {pred_label}")

# =====================================
# METRICS
# =====================================

TP = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
TN = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
FP = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
FN = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

accuracy = (TP + TN) / len(y_true)

precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print("\n" + "="*50)
print("FINAL RESULTS (DEEPSAFE - VIDEO)")
print("="*50)

print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix")
print(f"[[{TN}, {FP}]")
print(f" [{FN}, {TP}]]")

# =====================================
# SAVE RESULTS
# =====================================

df_results = pd.DataFrame({
    "video": [r["video_name"] for _, r in df.iterrows()],
    "true": y_true,
    "pred": y_pred
})

df_results.to_csv("deepsafe_video_results.csv", index=False)

print("\nSaved: deepsafe_video_results.csv")

# =====================================
# PLOT
# =====================================

cm = [[TN, FP], [FN, TP]]

plt.figure(figsize=(6, 6))
plt.imshow(cm)

plt.xticks([0, 1], ["Real", "Fake"])
plt.yticks([0, 1], ["Real", "Fake"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i][j]), ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("DeepSafe Video Confusion Matrix")

plt.savefig("deepsafe_video_cm.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: deepsafe_video_cm.png")