import os
import cv2

def extract_frames(video_dir, output_dir, frame_interval=50):

    os.makedirs(output_dir, exist_ok=True)

    videos = sorted([
        v for v in os.listdir(video_dir)
        if v.endswith(".mp4")
    ])

    total_saved = 0

    for video in videos:

        video_path = os.path.join(video_dir, video)

        cap = cv2.VideoCapture(video_path)

        frame_num = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_num % frame_interval == 0:

                filename = (
                    f"{os.path.basename(video_dir)}_"
                    f"{video[:-4]}_"
                    f"{frame_num}.jpg"
                )

                cv2.imwrite(
                    os.path.join(output_dir, filename),
                    frame
                )

                total_saved += 1

            frame_num += 1

        cap.release()

    print(
        f"{os.path.basename(video_dir)} "
        f"saved {total_saved} images"
    )