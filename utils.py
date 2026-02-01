import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def make_point_cloud(n=3000, seed=0):
    rng = np.random.default_rng(seed)

    # X,Y spread
    x = rng.normal(0.0, 1.0, n)
    y = rng.normal(0.0, 1.0, n)

    # Positive depth (Z) with a wide range + structure
    z = rng.gamma(shape=2.0, scale=2.0, size=n)  # mostly > 0, long tail

    # Add a slight "tilt" so depth changes with x,y (looks more 3D)
    z = z + 0.5 * x + 0.2 * y

    # Keep only points with z>0
    m = z > 0.1
    return np.stack([x[m], y[m], z[m]], axis=1)



def show_plotly_3d(X, marker_size=2, clip_percentile=98):
    X = np.asarray(X, dtype=float)
    m = np.isfinite(X).all(axis=1)
    X = X[m]
    if X.shape[0] == 0:
        raise ValueError("No valid points to plot.")

    # --- Robust outlier clipping (very important for triangulated clouds)
    # Clip by Z and also by radial distance to kill absurd far points
    z = X[:, 2]
    z_lo = np.percentile(z, 1)
    z_hi = np.percentile(z, clip_percentile)

    r = np.linalg.norm(X, axis=1)
    r_hi = np.percentile(r, clip_percentile)

    keep = (z >= z_lo) & (z <= z_hi) & (r <= r_hi)
    Xp = X[keep]
    if Xp.shape[0] < 50:  # fallback if too aggressive
        Xp = X

    # --- Axis ranges (robust)
    def prange(v, lo=1, hi=99, pad=0.05):
        a = np.percentile(v, lo)
        b = np.percentile(v, hi)
        span = max(b - a, 1e-9)
        p = span * pad
        return [a - p, b + p]

    xr = prange(Xp[:, 0], 1, 99)
    yr = prange(Xp[:, 1], 1, 99)
    zr = prange(Xp[:, 2], 1, 99)

    # --- Color by depth (Z)
    zc = Xp[:, 2]

    fig = go.Figure(data=[go.Scatter3d(
        x=Xp[:, 0],
        y=Xp[:, 1],
        z=Xp[:, 2],
        mode="markers",
        marker=dict(
            size=marker_size,
            color=zc,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Depth (Z)")
        )
    )])

    # --- Better default camera: looking toward the cloud from "above and back"
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.6, y=1.2, z=1.2)  # adjust if you want closer/farther
    )

    fig.update_layout(
        title=f"Interactive 3D point cloud (showing {Xp.shape[0]}/{X.shape[0]} points after clipping)",
        scene=dict(
            xaxis=dict(title="X", range=xr, backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="Y", range=yr, backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(title="Z", range=zr, backgroundcolor="rgba(0,0,0,0)"),
            aspectmode="data",
            camera=camera
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False
    )

    fig.show()


def show_plotly_3d_old(X):
    # Color by depth (Z)
    z = X[:, 2]

    fig = go.Figure(data=[go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        mode="markers",
        marker=dict(
            size=2,
            color=z,            # depth coloring
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Depth (Z)")
        )
    )])

    fig.update_layout(
        title="Interactive 3D point cloud (rotate / zoom / pan)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.show()



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


def draw_pairs(img_first, img_second, p0, p1, is_horizontal, max_lines=3000):
    # Make sure points are int
    p0 = p0.astype(int)
    p1 = p1.astype(int)

    h1, w1 = img_first.shape[:2]
    h2, w2 = img_second.shape[:2]

    #horizontal:
    if is_horizontal:
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img_first
        canvas[:h2, w1:w1 + w2] = img_second

    #vertical:
    else:
        canvas = np.zeros((h1 + h2, max(w1, w2), 3), dtype=np.uint8)
        canvas[:h1, :w1] = img_first
        canvas[h1:h1 + h2, :w2] = img_second

    # Optional: limit number of lines for clarity
    n = min(len(p0), max_lines)

    for i in range(n):
        pt1 = tuple(p0[i])
        if is_horizontal:
            pt2 = (p1[i][0] + w1, p1[i][1])  # horizontal
        else:
            pt2 = (p1[i][0], p1[i][1] + h1) #vertical

        cv2.circle(canvas, pt1, 3, (0, 255, 0), -1)
        cv2.circle(canvas, pt2, 3, (0, 255, 0), -1)
        cv2.line(canvas, pt1, pt2, (255, 0, 0), 1)

    show_bgr("Optical Flow Matches", canvas)
    #cv2.imshow("Optical Flow Matches", canvas)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return


def draw_epipolar_lines(E, K, p0i, p1i, img_second, stride=50):
    Kinv = np.linalg.inv(K)
    F = Kinv.T @ E @ Kinv

    lines2 = cv2.computeCorrespondEpilines(p0i.reshape(-1, 1, 2).astype(np.float32), 1, F).reshape(-1, 3)
    # Each line: a*x + b*y + c = 0 in image2 pixels

    img2_show = img_second.copy()
    #for (a, b, c), pt2 in zip(lines2, p1i):  # take every 50th to reduce clutter
    for (a, b, c), pt2 in zip(lines2[::stride], p1i[::stride]):  # take every 50th to reduce clutter
        x0, x1 = 0, img2_show.shape[1] - 1
        y0 = int((-c - a * x0) / b)
        y1 = int((-c - a * x1) / b)
        cv2.line(img2_show, (x0, y0), (x1, y1), (0, 255, 0), 1)
        cv2.circle(img2_show, tuple(pt2.astype(int)), 3, (0, 0, 255), -1)
    show_bgr("epipolar lines", img2_show)

def sampson_error(E, p0, p1, K):
    #convert to normalize camera coordinates
    p0n = cv2.undistortPoints(p0.reshape(-1,1,2), K, None).reshape(-1,2)
    p1n = cv2.undistortPoints(p1.reshape(-1,1,2), K, None).reshape(-1,2)

    #Homogeneous coordinates
    x1 = np.hstack([p0n, np.ones((len(p0n),1))])
    x2 = np.hstack([p1n, np.ones((len(p1n),1))])

    Ex1  = x1 @ E.T #epipolar line of x1 in image 2
    Etx2 = x2 @ E   #epipolar line of x2 in image 1

    x2tEx1 = np.sum(x2 * Ex1, axis=1) #Epipolar constraint error

    #(x2ᵀ E x1)² / (||∂(x2ᵀEx1)/∂x||²)
    #first-order approximation of geometric reprojection error
    denom = Ex1[:,0]**2 + Ex1[:,1]**2 + Etx2[:,0]**2 + Etx2[:,1]**2
    return (x2tEx1**2) / denom