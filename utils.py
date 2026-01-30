import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

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

def show_bgr(title, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 6))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def draw_pairs(img_first, img_second, p0, p1, max_lines=3000):
    # Make sure points are int
    p0 = p0.astype(int)
    p1 = p1.astype(int)

    h1, w1 = img_first.shape[:2]
    h2, w2 = img_second.shape[:2]

    #horizontal:
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img_first
    canvas[:h2, w1:w1 + w2] = img_second

    #vertical:
    #canvas = np.zeros((h1 + h2, max(w1, w2), 3), dtype=np.uint8)
    #canvas[:h1, :w1] = img_first
    #canvas[h1:h1 + h2, :w2] = img_second

    # Optional: limit number of lines for clarity
    n = min(len(p0), max_lines)

    for i in range(n):
        pt1 = tuple(p0[i])
        #pt2 = (p1[i][0] + w1, p1[i][1]) #horizontal
        pt2 = (p1[i][0], p1[i][1] + h1) #vertical

        cv2.circle(canvas, pt1, 3, (0, 255, 0), -1)
        cv2.circle(canvas, pt2, 3, (0, 255, 0), -1)
        cv2.line(canvas, pt1, pt2, (255, 0, 0), 1)

    show_bgr("Optical Flow Matches", canvas)
    #cv2.imshow("Optical Flow Matches", canvas)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return