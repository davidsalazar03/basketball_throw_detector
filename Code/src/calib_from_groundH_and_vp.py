"""
calib_from_groundH_and_vp.py

Pipeline (course method):
1) Estimate the ground-plane homography H (world XY, Z=0 -> image) with RANSAC
2) Report per-point errors + inliers/outliers and save the lists
3) Estimate intrinsics K from orthogonal vanishing points:
   - v_x ~ h1, v_y ~ h2 (columns of H)
   - v_z is read from a file (vertical vanishing point)
   Assumptions:
     skew s = 0, f_x = f_y = f, (c_x, c_y) = image center
4) Decompose H = K [r1 r2 t] to obtain R, t (world -> camera pose)
5) Validate by reprojecting the inliers (RMSE and per-point errors)
6) Save results

Requirements: numpy, pandas, opencv-python
"""

from __future__ import annotations

from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
import cv2


# ---------------- CONFIG ----------------
BASE_DIR = Path("inputs")         # input folder
OUT_DIR = Path("outputs")         # output folder

WORLD_CSV = BASE_DIR / "world_points_coord.csv"   # id,x,y,z
IMAGE_CSV = BASE_DIR / "image_points.csv"         # id,u,v
IMAGE_SIZE_TXT = BASE_DIR / "image_size.txt"      # "W H"
VP_VERTICAL_TXT = BASE_DIR / "vp_vertical.txt"    # "u,v,w" or "u v w"

# RANSAC
RANSAC_THRESH_PX = 3.0
RANSAC_MAX_ITERS = 5000
RANSAC_CONF = 0.999

# Outputs
OUT_DIR.mkdir(parents=True, exist_ok=True)
H_RANSAC_PATH = OUT_DIR / "H_ground_ransac.txt"
H_CLEAN_PATH = OUT_DIR / "H_ground_clean.txt"
INLIERS_PATH = OUT_DIR / "inliers_ids.txt"
OUTLIERS_PATH = OUT_DIR / "outliers_ids.txt"
POINT_ERRORS_CSV = OUT_DIR / "ground_point_errors.csv"
CALIB_JSON = OUT_DIR / "calibration_ground_vp.json"
CALIB_NPZ = OUT_DIR / "calibration_ground_vp.npz"
# --------------------------------------


def read_image_size(path: Path) -> tuple[int, int]:
    txt = path.read_text(encoding="utf-8").strip()
    parts = txt.split()
    if len(parts) != 2:
        raise ValueError(f"Invalid format in {path}. Expected: 'W H'.")
    W, H = int(parts[0]), int(parts[1])
    return W, H


def read_vp_vertical(path: Path) -> np.ndarray:
    txt = path.read_text(encoding="utf-8").strip()
    # accept "u,v,w" or "u v w"
    parts = re.split(r"[,\s]+", txt)
    parts = [p for p in parts if p != ""]
    if len(parts) != 3:
        raise ValueError(f"Invalid format in {path}. Expected: 3 numbers (u,v,w).")
    v = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)
    # normalize to w=1 (if possible)
    if abs(v[2]) > 1e-12:
        v = v / v[2]
    return v


def load_correspondences_ground(world_csv: Path, image_csv: Path, eps_z: float = 1e-6):
    wdf = pd.read_csv(world_csv)
    idf = pd.read_csv(image_csv)

    for col in ["id", "x", "y"]:
        if col not in wdf.columns:
            raise ValueError(f"{world_csv} must contain column '{col}'.")
    if "z" not in wdf.columns:
        wdf["z"] = 0.0

    for col in ["id", "u", "v"]:
        if col not in idf.columns:
            raise ValueError(f"{image_csv} must contain column '{col}'.")

    wdf["id"] = wdf["id"].astype(str).str.strip()
    idf["id"] = idf["id"].astype(str).str.strip()

    merged = wdf.merge(idf, on="id", how="inner")

    # ground only: z ~ 0
    merged = merged[np.abs(merged["z"].astype(float)) <= eps_z].copy()
    if len(merged) < 4:
        raise RuntimeError(f"Fewer than 4 ground-plane points (Z=0) with correspondences. N={len(merged)}.")

    ids = merged["id"].tolist()
    XY = merged[["x", "y"]].to_numpy(dtype=float)
    uv = merged[["u", "v"]].to_numpy(dtype=float)
    return ids, XY, uv


def apply_homography(H: np.ndarray, XY: np.ndarray) -> np.ndarray:
    pts = np.asarray(XY, dtype=float)
    pts_h = np.c_[pts, np.ones((pts.shape[0], 1), dtype=float)]
    q = (H @ pts_h.T).T
    q = q[:, :2] / (q[:, 2:3] + 1e-12)
    return q


def estimate_H_ransac(XY: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H, mask = cv2.findHomography(
        XY.astype(np.float64),
        uv.astype(np.float64),
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESH_PX,
        maxIters=RANSAC_MAX_ITERS,
        confidence=RANSAC_CONF,
    )
    if H is None or mask is None:
        raise RuntimeError("cv2.findHomography failed.")
    mask = mask.reshape(-1).astype(bool)
    # normalize H to H[2,2]=1 (when possible)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H, mask


def save_matrix_txt(path: Path, M: np.ndarray):
    lines = []
    for r in range(M.shape[0]):
        lines.append(" ".join(f"{M[r, c]:.12g}" for c in range(M.shape[1])))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def estimate_f_from_orthogonal_vps(vps: list[np.ndarray], W: int, H: int) -> tuple[float, tuple[float, float], list[float]]:
    """
    Assumptions:
      skew=0, fx=fy=f, (cx,cy) at the image center.
    For two orthogonal vanishing points v_i, v_j:
      (v_i - c)·(v_j - c) + f^2 = 0  =>  f^2 = - (v_i-c)·(v_j-c)

    vps: list [v_x, v_y, v_z] (each with w=1)
    returns f, (cx,cy), and the list of f^2 values computed for each pair
    """
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    c = np.array([cx, cy], dtype=float)

    f2_list = []
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        vi = vps[i][:2] - c
        vj = vps[j][:2] - c
        f2 = -float(np.dot(vi, vj))
        f2_list.append(f2)

    # keep positive values (negative indicates strong noise / inconsistency)
    f2_pos = [x for x in f2_list if x > 0]
    if len(f2_pos) == 0:
        raise RuntimeError(
            "Could not obtain a positive f^2 from the VPs (with cx,cy at the center and fx=fy). "
            "This suggests strong noise in the vertical VP or in H, or a mismatch with the center/assumptions."
        )

    # robust choice: median
    f2_med = float(np.median(np.array(f2_pos)))
    f = float(np.sqrt(f2_med))
    return f, (cx, cy), f2_list


def decompose_H_to_pose(K: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given H = K [r1 r2 t] (up to scale), recover R,t (world -> camera).
    """
    Kinv = np.linalg.inv(K)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]

    lam = 1.0 / (np.linalg.norm(Kinv @ h1) + 1e-12)

    r1 = lam * (Kinv @ h1)
    r2 = lam * (Kinv @ h2)
    r3 = np.cross(r1, r2)
    t = lam * (Kinv @ h3)

    R = np.column_stack([r1, r2, r3])

    # enforce R in SO(3) via SVD
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return R, t


def project_ground_points(K: np.ndarray, R: np.ndarray, t: np.ndarray, XY: np.ndarray) -> np.ndarray:
    """
    Project ground-plane points (Z=0): Xw=[X,Y,0]^T
    """
    Xw = np.c_[XY, np.zeros((XY.shape[0], 1), dtype=float)]  # Nx3
    Xc = (R @ Xw.T).T + t.reshape(1, 3)
    x = (K @ Xc.T).T
    uv = x[:, :2] / (x[:, 2:3] + 1e-12)
    return uv


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def main():
    W, Himg = read_image_size(IMAGE_SIZE_TXT)
    v_z = read_vp_vertical(VP_VERTICAL_TXT)

    ids, XY, uv = load_correspondences_ground(WORLD_CSV, IMAGE_CSV)

    # 1) Estimate H with RANSAC
    H_ransac, inlier_mask = estimate_H_ransac(XY, uv)

    inlier_ids = [pid for pid, m in zip(ids, inlier_mask) if m]
    outlier_ids = [pid for pid, m in zip(ids, inlier_mask) if not m]

    save_matrix_txt(H_RANSAC_PATH, H_ransac)
    INLIERS_PATH.write_text("\n".join(inlier_ids) + ("\n" if len(inlier_ids) else ""), encoding="utf-8")
    OUTLIERS_PATH.write_text("\n".join(outlier_ids) + ("\n" if len(outlier_ids) else ""), encoding="utf-8")

    # 2) Re-estimate a "clean" H using inliers only (LS, no RANSAC)
    XY_in = XY[inlier_mask]
    uv_in = uv[inlier_mask]
    H_clean, _ = cv2.findHomography(XY_in.astype(np.float64), uv_in.astype(np.float64), method=0)
    if H_clean is None:
        raise RuntimeError("H re-estimation (inliers) failed.")
    if abs(H_clean[2, 2]) > 1e-12:
        H_clean = H_clean / H_clean[2, 2]
    save_matrix_txt(H_CLEAN_PATH, H_clean)

    # 3) Per-point errors (with H_clean) and diagnostic CSV
    uv_proj_all = apply_homography(H_clean, XY)
    err_all = np.linalg.norm(uv_proj_all - uv, axis=1)

    diag = pd.DataFrame({
        "id": ids,
        "inlier": inlier_mask.astype(int),
        "err_px": err_all,
        "u_obs": uv[:, 0],
        "v_obs": uv[:, 1],
        "u_proj": uv_proj_all[:, 0],
        "v_proj": uv_proj_all[:, 1],
        "X": XY[:, 0],
        "Y": XY[:, 1],
    })
    diag.to_csv(POINT_ERRORS_CSV, index=False)

    print(f"[INFO] Ground points used: {len(ids)}")
    print(f"[INFO] RANSAC inliers: {int(inlier_mask.sum())}/{len(ids)}")
    print(f"[INFO] Saved H_ransac: {H_RANSAC_PATH}")
    print(f"[INFO] Saved H_clean:  {H_CLEAN_PATH}")
    print(f"[INFO] Saved inliers:  {INLIERS_PATH}")
    print(f"[INFO] Saved outliers: {OUTLIERS_PATH}")
    print(f"[INFO] Saved diagnostics: {POINT_ERRORS_CSV}")

    # 4) Estimate K from orthogonal vanishing points
    # v_x ~ h1, v_y ~ h2, v_z from file
    h1 = H_clean[:, 0].copy()
    h2 = H_clean[:, 1].copy()
    v_x = h1 / (h1[2] + 1e-12)
    v_y = h2 / (h2[2] + 1e-12)

    f, (cx, cy), f2_list = estimate_f_from_orthogonal_vps([v_x, v_y, v_z], W, Himg)

    K = np.array([[f, 0.0, cx],
                  [0.0, f, cy],
                  [0.0, 0.0, 1.0]], dtype=float)

    print(f"[INFO] Estimated f (px): {f:.3f}")
    print(f"[INFO] Assumed principal point (cx,cy): ({cx:.1f},{cy:.1f})")
    print(f"[INFO] f^2 from VP pairs (vx-vy, vx-vz, vy-vz): {[float(x) for x in f2_list]}")

    # 5) Plane pose (world->camera) from H_clean and K
    R, t = decompose_H_to_pose(K, H_clean)

    # 6) Validate via reprojection (inliers only)
    uv_hat_in = project_ground_points(K, R, t, XY_in)
    rmse_in = rmse(uv_in, uv_hat_in)
    print(f"[CHECK] Reprojection RMSE (inliers, pinhole model): {rmse_in:.3f} px")

    # Camera center in world coordinates (optional)
    C = -R.T @ t

    # 7) Save results
    payload = {
        "image_size": {"width": int(W), "height": int(Himg)},
        "assumptions": {
            "skew": 0,
            "fx_equals_fy": True,
            "principal_point": "image_center",
        },
        "H_clean_world_to_image": H_clean.tolist(),
        "vp": {
            "vx_from_H": v_x.tolist(),
            "vy_from_H": v_y.tolist(),
            "vz_input": v_z.tolist(),
        },
        "K": K.tolist(),
        "R_world_to_cam": R.tolist(),
        "t_world_to_cam": t.tolist(),
        "C_cam_in_world": C.tolist(),
        "rmse_inliers_px": rmse_in,
        "ransac": {
            "threshold_px": RANSAC_THRESH_PX,
            "max_iters": RANSAC_MAX_ITERS,
            "confidence": RANSAC_CONF,
            "n_points": len(ids),
            "n_inliers": int(inlier_mask.sum()),
        }
    }
    CALIB_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    np.savez(
        CALIB_NPZ,
        K=K, R=R, t=t, C=C,
        H_clean=H_clean, v_x=v_x, v_y=v_y, v_z=v_z,
        image_size=np.array([W, Himg], dtype=int),
        ids=np.array(ids, dtype=object),
        inlier_mask=inlier_mask.astype(np.uint8),
        XY=XY, uv=uv
    )

    print(f"[DONE] Saved calibration: {CALIB_JSON}")
    print(f"[DONE] Saved calibration: {CALIB_NPZ}")


if __name__ == "__main__":
    main()
