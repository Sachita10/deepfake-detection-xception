from extract_frames import extract_frames
import os

real_video_dir = os.path.expanduser(
    "~/Downloads/FaceForensics++_C23/original"
)

extract_frames(
    video_dir=real_video_dir,
    output_dir="dataset/raw/real",
    frame_interval=50
)

fake_methods = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
    "FaceShifter",
    "DeepFakeDetection"
]

for method in fake_methods:

    video_dir = os.path.expanduser(
        f"~/Downloads/FaceForensics++_C23/{method}"
    )

    extract_frames(
        video_dir=video_dir,
        output_dir=f"dataset/raw/{method}",
        frame_interval=50
    )
import os
import random
import shutil

random.seed(42)

os.makedirs("dataset/balanced/real", exist_ok=True)
os.makedirs("dataset/balanced/fake", exist_ok=True)
real_imgs = os.listdir("dataset/raw/real")

for img in real_imgs:
    shutil.copy(
        os.path.join("dataset/raw/real", img),
        os.path.join("dataset/balanced/real", img)
    )

print("Real copied:", len(real_imgs))
methods = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
    "FaceShifter",
    "DeepFakeDetection"
]

target_per_method = 1781

for method in methods:

    source = f"dataset/raw/{method}"

    imgs = os.listdir(source)

    selected = random.sample(
        imgs,
        target_per_method
    )

    for img in selected:

        shutil.copy(
            os.path.join(source, img),
            os.path.join(
                "dataset/balanced/fake",
                f"{method}_{img}"
            )
        )

print("Fake balancing complete")
print(
    "Real:",
    len(os.listdir("dataset/balanced/real"))
)

print(
    "Fake:",
    len(os.listdir("dataset/balanced/fake"))
)
import os
import random
import shutil

random.seed(42)

classes = ["real", "fake"]

for cls in classes:

    source = f"dataset/balanced/{cls}"

    train_dir = f"dataset/train/{cls}"
    val_dir = f"dataset/val/{cls}"
    test_dir = f"dataset/test/{cls}"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    images = os.listdir(source)

    random.shuffle(images)

    total = len(images)

    train_size = int(0.70 * total)
    val_size = int(0.15 * total)

    train_imgs = images[:train_size]

    val_imgs = images[
        train_size:train_size + val_size
    ]

    test_imgs = images[
        train_size + val_size:
    ]

    for img in train_imgs:
        shutil.copy(
            os.path.join(source, img),
            os.path.join(train_dir, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(source, img),
            os.path.join(val_dir, img)
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(source, img),
            os.path.join(test_dir, img)
        )

print("Dataset split complete!")
for split in ["train", "val", "test"]:

    real = len(os.listdir(f"dataset/{split}/real"))
    fake = len(os.listdir(f"dataset/{split}/fake"))

    print(f"\n{split.upper()}")
    print("Real:", real)
    print("Fake:", fake)