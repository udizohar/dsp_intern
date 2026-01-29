import cv2
import os
import numpy as np

def load_video(input_dir, video_name, output_dir):
    video_path = os.path.join(input_dir, video_name)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Resolution:", width, "x", height)
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


def get_motion_two_images(K, img_first, img_second):
    return 100, 200

if __name__ == '__main__':
    input_dir = "video_input"
    video_name = "1.mp4"
    output_dir = "frames"

    #load_video(input_dir, video_name, output_dir)
    image_width = 1920
    image_height = 1080
    focal_xy = 1620

    K = np.array([focal_xy, 0, image_width / 2,
                    0, focal_xy, image_height / 2,
                    0, 0, 1], dtype=np.float64)

    recording_folder_name = "1"
    folder_path = os.path.join(output_dir, recording_folder_name)
    first_frame_idx = 0
    second_frame_idx = 1
    first_image_path = os.path.join(folder_path, f"frame_{first_frame_idx:06d}.png")
    second_image_path = os.path.join(folder_path, f"frame_{second_frame_idx:06d}.png")
    img_first = cv2.imread(first_image_path)
    img_second = cv2.imread(second_image_path)

    x, y = get_motion_two_images(K, img_first, img_second)

    print(x,y)