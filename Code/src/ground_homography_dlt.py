from pathlib import Path
import csv
import numpy as np
import cv2

# ---------------- CONFIG ----------------
WORLD_CSV = r"inputs/world_points_coord.csv"     # id,x,y,z (meters)
IMAGE_CSV = r"inputs/image_points.csv"           # id,u,v (pixels) (your output)

IMAGE_PATH = r"data/new_dataset/images/shot_1_f0000.jpg"
OUT_RECTIFIED = r"outputs/rectified_floor.png"

PX_PER_M = 120   # rectification scale (e.g., 120 px per meter) -> adjust as needed
MARGIN_M = 0.5   # margin (in meters) around the points' bounding box
USE_RANSAC = True
# ---------------------------------------


def read_world_points(path):
    pts = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = row["id"].strip()
            x = float(row["x"])
            y = float(row["y"])
            z = float(row.get("z", 0.0))  # if missing, assume 0
            pts[pid] = (x, y, z)
    return pts


def read_image_points(path):
    pts = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = row["id"].strip()
            if row["u"] == "" or row["v"] == "":
                continue
            u = float(row["u"])
            v = float(row["v"])
            pts[pid] = (u, v)
    return pts


def normalize_points_2d(pts):
    """
    pts: (N,2)
    returns T (3x3) and pts_norm (N,2) such that x_norm ~ T x
    """
    pts = np.asarray(pts, dtype=float)
    mean = pts.mean(axis=0)
    d = np.sqrt(((pts - mean) ** 2).sum(axis=1)).mean()
    if d < 1e-12:
        raise ValueError("Degenerate points in normalization.")

    s = np.sqrt(2) / d
    T = np.array([[s, 0, -s * mean[0]],
                  [0, s, -s * mean[1]],
                  [0, 0, 1]], dtype=float)

    pts_h = np.c_[pts, np.ones((pts.shape[0], 1))]
    pts_n = (T @ pts_h.T).T
    pts_n = pts_n[:, :2] / pts_n[:, 2:3]
    return T, pts_n


def dlt_homography(world_xy, image_uv):
    """
    Estimates H such that [u v 1]^T ~ H [X Y 1]^T
    via DLT with normalization (Hartley).
    world_xy: (N,2), image_uv: (N,2), N>=4
    """
    world_xy = np.asarray(world_xy, dtype=float)
    image_uv = np.asarray(image_uv, dtype=float)
    if world_xy.shape[0] < 4:
        raise ValueError("You need at least 4 ground-plane correspondences for a homography.")

    Tw, w_n = normalize_points_2d(world_xy)
    Ti, i_n = normalize_points_2d(image_uv)

    N = world_xy.shape[0]
    A = []
    for k in range(N):
        X, Y = w_n[k]
        u, v = i_n[k]
        # DLT rows
        A.append([0, 0, 0, -X, -Y, -1, v * X, v * Y, v])
        A.append([X, Y, 1,  0,  0,  0, -u * X, -u * Y, -u])
    A = np.asarray(A, dtype=float)

    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)

    # denormalize: i ~ Ti^{-1} Hn Tw
    H = np.linalg.inv(Ti) @ Hn @ Tw

    # normalize to H[2,2]=1 (if possible)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def apply_homography(H, pts):
    """
    H: 3x3, pts: (N,2)
    returns (N,2)
    """
    pts = np.asarray(pts, dtype=float)
    pts_h = np.c_[pts, np.ones((pts.shape[0], 1))]
    q = (H @ pts_h.T).T
    q = q[:, :2] / q[:, 2:3]
    return q


def main():
    world = read_world_points(WORLD_CSV)
    image = read_image_points(IMAGE_CSV)

    plane_type = "ground"  # CHANGE HERE: "ground" or "vertical"

    # keep only common IDs and filter by plane
    common = []
    world_coords = []  # will store (coord1, coord2) depending on the chosen plane
    image_uv = []

    if plane_type == "vertical":
        X0 = 6.58
        EPS = 1e-3

        for pid, (x, y, z) in world.items():
            if abs(x - X0) > EPS:
                continue
            if pid in image:
                u, v = image[pid]
                common.append(pid)
                # For a vertical plane X=X0, use (Y, Z)
                world_coords.append((y, z))
                image_uv.append((u, v))

        print(f"[INFO] VERTICAL plane (X={X0}m): {len(common)} correspondences")

    elif plane_type == "ground":
        EPS = 1e-3

        for pid, (x, y, z) in world.items():
            if abs(z) > EPS:
                continue
            if pid in image:
                u, v = image[pid]
                common.append(pid)
                # For the ground plane (Z=0), use (X, Y)
                world_coords.append((x, y))
                image_uv.append((u, v))

        print(f"[INFO] GROUND plane (Z=0): {len(common)} correspondences")

    else:
        raise ValueError(f"Unknown plane_type: {plane_type}")

    world_pts = np.asarray(world_coords, dtype=float)
    image_uv = np.asarray(image_uv, dtype=float)

    if len(common) < 4:
        raise RuntimeError(f"Not enough correspondences (>=4). Found: {len(common)}")

    # Display used points
    print(f"[DEBUG] Used IDs: {common}")
    print(f"[DEBUG] First 3 world->image correspondences:")
    for i in range(min(3, len(common))):
        print(f"  {common[i]}: world{world_pts[i]} -> image{image_uv[i]}")

    # -------- estimate H --------
    if USE_RANSAC:
        # cv2.findHomography estimates H (world->image) with RANSAC
        H_cv, mask = cv2.findHomography(world_pts, image_uv, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H_cv is None:
            raise RuntimeError("cv2.findHomography failed.")
        H = H_cv
        inliers = int(mask.sum()) if mask is not None else len(common)
        print(f"[INFO] RANSAC inliers: {inliers}/{len(common)}")
    else:
        H = dlt_homography(world_pts, image_uv)

    print(f"[DEBUG] Homography H (world->image):")
    print(H)

    Hinv = np.linalg.inv(H)

    # -------- rectify the plane into an output image --------
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Could not read: {IMAGE_PATH}")

    # world bounding box to define the metric canvas
    coord_min = world_pts.min(axis=0) - MARGIN_M
    coord_max = world_pts.max(axis=0) + MARGIN_M

    print(f"[DEBUG] World bounding box: min={coord_min}, max={coord_max}")

    out_w = int(np.ceil((coord_max[0] - coord_min[0]) * PX_PER_M))
    out_h = int(np.ceil((coord_max[1] - coord_min[1]) * PX_PER_M))

    print(f"[INFO] Output canvas: {out_w} x {out_h} pixels")

    # Mapping "world->canvas_px"
    # For both planes, we want:
    #   - horizontal coordinate (coord1) increases to the right: x_px = (coord1 - min1) * PX_PER_M
    #   - vertical coordinate (coord2) increases upward in the world, but pixels increase downward
    #     so: y_px = (max2 - coord2) * PX_PER_M
    T = np.array([
        [PX_PER_M,  0,          -coord_min[0] * PX_PER_M],
        [0,         -PX_PER_M,  coord_max[1] * PX_PER_M],
        [0,          0,          1]
    ], dtype=float)

    print(f"[DEBUG] World->canvas transform T:")
    print(T)

    # We want warpPerspective using a matrix that maps image -> canvas.
    # image -> world: Hinv
    # world -> canvas: T
    # therefore: image -> canvas = T @ Hinv
    W = T @ Hinv

    print(f"[DEBUG] Final warp matrix W (image->canvas):")
    print(W)

    rect = cv2.warpPerspective(img, W, (out_w, out_h))
    Path(OUT_RECTIFIED).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(OUT_RECTIFIED, rect)
    print(f"[DONE] Rectification saved to: {OUT_RECTIFIED}")

    # -------- sanity check (mean reprojection error) --------
    proj_uv = apply_homography(H, world_pts)  # world->image
    err = np.linalg.norm(proj_uv - image_uv, axis=1)
    print(f"[CHECK] Reprojection error: mean={err.mean():.2f}px | max={err.max():.2f}px")

    # Show individual errors for debugging
    print(f"[DEBUG] Individual reprojection errors:")
    for i, pid in enumerate(common):
        print(f"  {pid}: {err[i]:.2f}px")

    H_GROUND_PATH = r"inputs/H_ground.txt"

    # Write (append to the file)
    with open(H_GROUND_PATH, "a", encoding="utf-8") as f:
        f.write(H[0].__str__() + "\n")
        f.write(H[1].__str__() + "\n")
        f.write(H[2].__str__() + "\n")
    print(f"[DONE] World->image homography matrix saved to: {H_GROUND_PATH}")


if __name__ == "__main__":
    main()
