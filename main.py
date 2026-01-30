import cv2
import os
import numpy as np

from utils import load_video, draw_pairs, show_bgr


def recover_pose_from_E_cheirality(E, p0, p1, K=None, dist=None, distance_thresh=np.inf):
    """
    p0,p1: Nx2 (pixels if K is not None, otherwise normalized)
    Returns: R_best, t_best, mask_best (Nx1 uint8), counts_per_candidate
    """
    p0 = np.asarray(p0, dtype=np.float64).reshape(-1,2)
    p1 = np.asarray(p1, dtype=np.float64).reshape(-1,2)

    # normalize if K is given
    if K is not None:
        p0n = cv2.undistortPoints(p0.reshape(-1,1,2), K, dist).reshape(-1,2)
        p1n = cv2.undistortPoints(p1.reshape(-1,1,2), K, dist).reshape(-1,2)
    else:
        p0n, p1n = p0, p1

    R1, R2, t = cv2.decomposeEssentialMat(E)
    cands = [(R1, t), (R1, -t), (R2, t), (R2, -t)]

    P0 = np.hstack([np.eye(3), np.zeros((3,1))])

    best_idx = -1
    best_count = -1
    best_mask = None
    best_R = None
    best_t = None
    counts = []

    for (R, tt) in cands:
        P1 = np.hstack([R, tt])

        X = cv2.triangulatePoints(P0, P1, p0n.T, p1n.T)   # 4xN homogeneous
        Z = X[2]
        W = X[3]

        depth0 = Z / W                          # z in cam0 after homog normalization
        Xeu = (X[:3] / W).T                     # Nx3 euclidean in cam0
        X1 = (R @ Xeu.T + tt).T                 # Nx3 in cam1
        depth1 = X1[:,2]

        good = (depth0 > 0) & (depth1 > 0) & (depth0 < distance_thresh) & (depth1 < distance_thresh)
        mask = good.astype(np.uint8).reshape(-1,1)
        cnt = int(np.count_nonzero(mask))
        counts.append(cnt)

        if cnt > best_count:
            best_count = cnt
            best_idx = len(counts)-1
            best_mask = mask
            best_R = R
            best_t = tt

    return best_R, best_t, best_mask, counts, best_idx


def sampson_error(E, p0, p1, K):
    p0n = cv2.undistortPoints(p0.reshape(-1,1,2), K, None).reshape(-1,2)
    p1n = cv2.undistortPoints(p1.reshape(-1,1,2), K, None).reshape(-1,2)

    x1 = np.hstack([p0n, np.ones((len(p0n),1))])
    x2 = np.hstack([p1n, np.ones((len(p1n),1))])

    Ex1  = x1 @ E.T
    Etx2 = x2 @ E

    x2tEx1 = np.sum(x2 * Ex1, axis=1)
    denom = Ex1[:,0]**2 + Ex1[:,1]**2 + Etx2[:,0]**2 + Etx2[:,1]**2
    return (x2tEx1**2) / denom

def draw_epipolar_lines(E, K, p0i, p1i, img_second):
    Kinv = np.linalg.inv(K)
    F = Kinv.T @ E @ Kinv

    lines2 = cv2.computeCorrespondEpilines(p0i.reshape(-1, 1, 2).astype(np.float32), 1, F).reshape(-1, 3)
    # Each line: a*x + b*y + c = 0 in image2 pixels

    img2_show = img_second.copy()
    #for (a, b, c), pt2 in zip(lines2, p1i):  # take every 50th to reduce clutter
    for (a, b, c), pt2 in zip(lines2[::50], p1i[::50]):  # take every 50th to reduce clutter
        x0, x1 = 0, img2_show.shape[1] - 1
        y0 = int((-c - a * x0) / b)
        y1 = int((-c - a * x1) / b)
        cv2.line(img2_show, (x0, y0), (x1, y1), (0, 255, 0), 1)
        cv2.circle(img2_show, tuple(pt2.astype(int)), 3, (0, 0, 255), -1)
    show_bgr("epipolar lines", img2_show)

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

    #without undistortion
    E, inliers = cv2.findEssentialMat(p0, p1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    inl = inliers.ravel().astype(bool)
    p0i = p0[inl].astype(np.float64)
    p1i = p1[inl].astype(np.float64)

    R, t, pose_mask, counts, idx = recover_pose_from_E_cheirality(E, p0i, p1i, K, dist=None, distance_thresh=1e6)

    U, S, Vt = np.linalg.svd(E)
    print("S:", S, "ratio s1/s2:", S[0] / S[1], "s3:", S[2])

    #If S[0]/S[1] is far from 1 or S[2] not small, you can “project” E to the closest essential:
    #E_proj = U @ np.diag([1, 1, 0]) @ Vt

    sampson_err = sampson_error(E, p0, p1, K)
    draw_epipolar_lines(E, K, p0i, p1i, img_second)

    #_, R_rel, t_rel, pose_inliers = cv2.recoverPose(E, p0i, p1i, K)



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

