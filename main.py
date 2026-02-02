from utils import *

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
    best_Xeu_xyz = None

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
            best_Xeu_xyz = Xeu

    return best_R, best_t, best_Xeu_xyz, best_mask, best_count, counts, best_idx



def get_motion_two_images(K, img_second, first_gray, second_gray, first_pts_all, lk_params,
                          map_Xw): #, map_uv_prev, is_bootstrap
    #Shi-Tomasi Detector in default
    #useHarrisDetector=True
    second_pts_all, status_all, optical_err_all = cv2.calcOpticalFlowPyrLK(first_gray, second_gray, first_pts_all, None, **lk_params)
    status_all = status_all.reshape(-1)

    R = np.eye(3, dtype=np.float64)
    t = np.zeros((3, 1), dtype=np.float64)
    traj = [t.copy()]


    p0_optical_status_ok = first_pts_all[status_all == 1].reshape(-1, 2)
    p1_optical_status_ok = second_pts_all[status_all == 1].reshape(-1, 2)
    err_optical_status_ok = optical_err_all[status_all == 1].reshape(-1)

    optical_flow_err_threshold = 10.0  # tune 10..30, low is a better keypoint
    low_error_bool = err_optical_status_ok < optical_flow_err_threshold

    p0_low_flow_err = p0_optical_status_ok[low_error_bool]
    p1_low_flow_err = p1_optical_status_ok[low_error_bool]

    E, inliers_e = cv2.findEssentialMat(p0_low_flow_err, p1_low_flow_err, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    inliers_e_bool = inliers_e.ravel().astype(bool)
    p0_inliers_e_bool = p0_low_flow_err[inliers_e_bool].astype(np.float64)
    p1_inliers_e_bool = p1_low_flow_err[inliers_e_bool].astype(np.float64)

    R, t, Xue_xyz, pose_mask_int, best_count, counts, idx = recover_pose_from_E_cheirality(E, p0_inliers_e_bool, p1_inliers_e_bool, K, dist=None, distance_thresh=1e6)
    pose_mask_bool = pose_mask_int.ravel().astype(bool)
    Xue_xyz_inliers = Xue_xyz[pose_mask_bool]

    p0_inliers_pose = p0_inliers_e_bool[pose_mask_bool]
    p1_inliers_pose = p1_inliers_e_bool[pose_mask_bool]


    #Xue_xyz_inliers_all.extend(Xue_xyz_inliers) # X should be fixed with accumulated t ! something like X+=t_total
    show_plotly_3d(Xue_xyz_inliers)


    #additional E validation:
    U, S, Vt = np.linalg.svd(E)
    print("S:", S, "ratio s1/s2:", S[0] / S[1], "s3:", S[2])
    #If S[0]/S[1] is far from 1 or S[2] not small, you can “project” E to the closest essential:
    #E_proj = U @ np.diag([1, 1, 0]) @ Vt

    sampson_err = sampson_error(E, p0_inliers_pose, p1_inliers_pose, K)
    sampson_err_threshold = 1e-4
    sampson_count = np.count_nonzero(sampson_err > sampson_err_threshold)
    sampson_percent = 100.0 * sampson_count / sampson_err.size
    print("Epipolar constraint error (Sampson approximation) values above threshold percent = ", sampson_percent)

    draw_epipolar_lines(E, K, p0_inliers_pose, p1_inliers_pose, img_second, stride=5)

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

    #draw_pairs(img_first, img_second, p0_inliers_e_bool, p1_inliers_e_bool, is_horizontal=False)
    #draw_pairs(img_first, img_second, p0_inliers_e_bool, p1_inliers_e_bool, is_horizontal=True)
    return second_pts_all, map_Xw



if __name__ == '__main__':
    #X = make_point_cloud(n=3000, seed=15)
    #show_plotly_3d(X)



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
    cv2.setRNGSeed(0)  # cosntant findEssentialMat randomness for debugging

    images_count = 163
    frames_stride = 10

    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    is_first_entry = True
    first_pts_all = None
    Xue_xyz_inliers_all = []

    for first_frame_idx_base in range(int(images_count / frames_stride)):
        first_frame_idx = first_frame_idx_base * frames_stride
        second_frame_idx = first_frame_idx + frames_stride
        if (first_frame_idx >= images_count):
            break

        first_image_path = os.path.join(folder_path, f"frame_{first_frame_idx:06d}.png")
        second_image_path = os.path.join(folder_path, f"frame_{second_frame_idx:06d}.png")
        img_first = cv2.imread(first_image_path)
        img_second = cv2.imread(second_image_path)

        first_gray = cv2.cvtColor(img_first, cv2.COLOR_BGR2GRAY)
        second_gray = cv2.cvtColor(img_second, cv2.COLOR_BGR2GRAY)

        if is_first_entry:
            first_pts_all = cv2.goodFeaturesToTrack(first_gray, maxCorners=3000, qualityLevel=0.01, minDistance=7)
            is_first_entry = False

        second_pts_all, Xue_xyz_inliers_all = get_motion_two_images(K, img_second, first_gray, second_gray, first_pts_all, lk_params, Xue_xyz_inliers_all)
        first_pts_all = second_pts_all

