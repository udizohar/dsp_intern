import cv2
import os

def load_video(input_dir, video_name, output_dir):
    video_path = os.path.join(input_dir, video_name)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else None
    print("fps = ", fps)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1

    cap.release()
    print(f"Saved {frame_idx} frames")


if __name__ == '__main__':
    input_dir = "video_input"
    video_name = "1.mp4"
    output_dir = "frames"

    load_video(input_dir, video_name, output_dir)
