import cv2
import os
import numpy as np

from utils import load_video, draw_pairs

def get_motion_two_images(K, img_first, img_second):
    first_gray = cv2.cvtColor(img_first, cv2.COLOR_BGR2GRAY)
    #Shi-Tomasi Detector in default
    #useHarrisDetector=True
    first_pts = cv2.goodFeaturesToTrack(first_gray, maxCorners=3000, qualityLevel=0.01, minDistance=7)

    R = np.eye(3, dtype=np.float64)
    t = np.zeros((3, 1), dtype=np.float64)

    traj = [t.copy()]

    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    second_gray = cv2.cvtColor(img_second, cv2.COLOR_BGR2GRAY)

    # Track points
    #second_pts, status, err = cv2.calcOpticalFlowPyrLK(second_gray, first_gray, first_pts, None, **lk_params)
    second_pts, status, err = cv2.calcOpticalFlowPyrLK(first_gray, second_gray, first_pts, None, **lk_params)
    status = status.reshape(-1)

    p0 = first_pts[status == 1].reshape(-1, 2)
    p1 = second_pts[status == 1].reshape(-1, 2)
    e = err[status == 1].reshape(-1)

    good = e < 20.0  # tune 10..30

    p0 = p0[good]
    p1 = p1[good]


    # Essential matrix with RANSAC

    #with undistortion
    p0n = cv2.undistortPoints(p0.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    p1n = cv2.undistortPoints(p1.reshape(-1, 1, 2), K, None).reshape(-1, 2)

    E, inliers = cv2.findEssentialMat(p0n, p1n, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=1e-3)
    inl = inliers.ravel().astype(bool)
    p0ni = p0n[inl].astype(np.float64)
    p1ni = p1n[inl].astype(np.float64)
    _, R_rel, t_rel, pose_inliers = cv2.recoverPose(E, p0ni, p1ni, K)


    #without undistortion
    E, inliers = cv2.findEssentialMat(p0, p1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    inl = inliers.ravel().astype(bool)
    p0i = p0[inl].astype(np.float64)
    p1i = p1[inl].astype(np.float64)
    _, R_rel, t_rel, pose_inliers = cv2.recoverPose(E, p0i, p1i, K)

    if E is None:
        return 0,0


    # Recover pose
    #_, R_rel, t_rel, pose_inliers = cv2.recoverPose(E, p0, p1, K, mask=inliers)

    # IMPORTANT: t_rel has unknown scale. You can set scale=1, or estimate scale externally.
    scale = 1.0


    # Accumulate (this is one consistent way; depends on your chosen convention)
    '''
    #missing part:
    t = t + scale * (R @ t_rel)
    R = R_rel @ R

    traj.append(t.copy())

    # Prepare next iteration
    prev_gray = gray
    prev_pts = p1.reshape(-1, 1, 2).astype(np.float32)
    '''
    return p0, p1



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
                    0, 0, 1], dtype=np.float64).reshape(3, 3)

    recording_folder_name = "1"
    folder_path = os.path.join(output_dir, recording_folder_name)
    first_frame_idx = 0
    second_frame_idx = 1 # !!!!!!!!!!!!!!!!!!!
    first_image_path = os.path.join(folder_path, f"frame_{first_frame_idx:06d}.png")
    second_image_path = os.path.join(folder_path, f"frame_{second_frame_idx:06d}.png")
    img_first = cv2.imread(first_image_path)
    img_second = cv2.imread(second_image_path)

    p0, p1 = get_motion_two_images(K, img_first, img_second)
    draw_pairs(img_first, img_second, p0, p1, is_horizontal=False)

