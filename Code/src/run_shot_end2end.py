import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ---------------- USER PATHS ----------------
VIDEO_PATH = r"data/new_videos_raw/shot_9.mp4"
MODEL_PATH = r"runs/detect/train3/weights/best.pt"

CALIB_JSON = r"outputs/calibration_refined.json"
KEYPOINTS_WORLD_CSV = r"inputs/keypoints_world.csv"
KEYPOINTS_IMAGE_CSV = r"inputs/keypoints_image.csv"

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_VIDEO = OUT_DIR / "overlay_shot_4.mp4"
OUT_DETECTIONS = OUT_DIR / "ball_detections.csv"
OUT_TRAJ_WORLD = OUT_DIR / "ball_trajectory_world.csv"
OUT_DECISION = OUT_DIR / "shot_decision.json"
# -------------------------------------------

# Basket parameters
HOOP_DIAM_M = 0.45
HOOP_R_M = HOOP_DIAM_M / 2.0
HOOP_Z_OFFSET_M = -0.08  # adjust: -0.10 lowers hoop by 10 cm

BALL_R_M = 0.12
ENTER_N_CONSEC = 3       # number of consecutive frames (2 or 3 is typical)
ENTER_MARGIN_M = 0.00    # extra margin (set 0.01 if you want to be more permissive)

# Decision (simple and not conservative)
TOL_M = 0.05  # extra tolerance (m): tune if needed

# Detection
CONF_MIN = 0.25
BALL_CLASS_ID = None  # set the class id if your model has multiple classes

# Overlay
MAX_TRAIL = 140


def load_calibration(calib_json: str):
    data = json.loads(Path(calib_json).read_text(encoding="utf-8"))

    K = np.array(data["refined"]["K"], dtype=float)
    dist = np.array(data["refined"]["dist"], dtype=float).reshape(-1)

    dist_cv = np.zeros((5, 1), dtype=float)
    for i in range(min(5, dist.size)):
        dist_cv[i, 0] = float(dist[i])

    R0 = np.array(data["refined"]["R_world_to_cam"], dtype=float)
    t0 = np.array(data["refined"]["t_world_to_cam"], dtype=float).reshape(3, 1)
    rvec0, _ = cv2.Rodrigues(R0)

    return {"K": K, "dist": dist_cv, "rvec": rvec0, "t": t0, "raw": data}


def load_keypoints(world_csv: str, image_csv: str):
    w = pd.read_csv(world_csv)
    i = pd.read_csv(image_csv)
    w["id"] = w["id"].astype(str).str.strip()
    i["id"] = i["id"].astype(str).str.strip()

    w_map = {r["id"]: np.array([r["x"], r["y"], r["z"]], dtype=float) for _, r in w.iterrows()}
    i_map = {r["id"]: np.array([r["u"], r["v"]], dtype=float) for _, r in i.iterrows()}

    common = sorted(set(w_map.keys()) & set(i_map.keys()))
    if len(common) < 4:
        raise ValueError(f"You need >=4 3D-2D correspondences. Found {len(common)}: {common}")

    obj = np.stack([w_map[k] for k in common], axis=0).astype(np.float64)  # Nx3
    img = np.stack([i_map[k] for k in common], axis=0).astype(np.float64)  # Nx2
    return common, obj, img, w_map


def refine_pose_solvepnp(calib: dict, obj_pts: np.ndarray, img_pts: np.ndarray):
    K = calib["K"]
    dist = calib["dist"]
    rvec0 = calib["rvec"].copy()
    t0 = calib["t"].copy()

    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=obj_pts,
        imagePoints=img_pts,
        cameraMatrix=K,
        distCoeffs=dist,
        rvec=rvec0,
        tvec=t0,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise RuntimeError("solvePnP failed.")

    # Reprojection RMSE
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - img_pts, axis=1)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return rvec, tvec, rmse


def undistort_to_normalized(u, v, K, dist):
    pts = np.array([[[u, v]]], dtype=np.float64)
    und = cv2.undistortPoints(pts, K, dist, P=None)
    x_n, y_n = und.reshape(2)
    return np.array([x_n, y_n, 1.0], dtype=float)


def ray_world_from_pixel(u, v, K, dist, R_wc):
    d_c = undistort_to_normalized(u, v, K, dist)   # ray direction in camera frame
    d_w = R_wc @ d_c                               # ray direction in world frame
    d_w = d_w / (np.linalg.norm(d_w) + 1e-12)
    return d_w


def intersect_ray_plane_X(C, d_w, X0):
    dx = float(d_w[0])
    if abs(dx) < 1e-12:
        return None
    s = (X0 - float(C[0])) / dx
    if s <= 0:
        return None
    return C + s * d_w


def choose_ball_detection_ultralytics(result, ball_class_id=None):
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    idxs = np.arange(len(conf))
    if ball_class_id is not None:
        idxs = idxs[cls == int(ball_class_id)]
        if idxs.size == 0:
            return None

    best = idxs[np.argmax(conf[idxs])]
    x1, y1, x2, y2 = xyxy[best]
    return float(x1), float(y1), float(x2), float(y2), float(conf[best])


def draw_hoop_polyline(overlay, K, dist, rvec, tvec, V3_world):
    thetas = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    pts = []
    for th in thetas:
        Pw = V3_world + np.array([HOOP_R_M * np.cos(th), HOOP_R_M * np.sin(th), 0.0], dtype=float)
        uv, _ = cv2.projectPoints(Pw.reshape(1, 3), rvec, tvec, K, dist)
        u, v = uv.reshape(2)
        if np.isfinite(u) and np.isfinite(v):
            pts.append([u, v])
    if len(pts) >= 6:
        pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], True, (255, 0, 0), 2, cv2.LINE_AA)


def main():
    from ultralytics import YOLO

    calib = load_calibration(CALIB_JSON)
    common, obj_pts, img_pts, w_map = load_keypoints(KEYPOINTS_WORLD_CSV, KEYPOINTS_IMAGE_CSV)

    # Refine pose
    rvec, tvec, rmse = refine_pose_solvepnp(calib, obj_pts, img_pts)
    print(f"[POSE] solvePnP RMSE (px): {rmse:.3f} | pts={common}")

    K = calib["K"]
    dist = calib["dist"]

    # Extract R_wc and camera center C in world coordinates
    R_cw, _ = cv2.Rodrigues(rvec)  # world->cam
    R_wc = R_cw.T
    C = -R_wc @ tvec.reshape(3)

    V3 = w_map["V3"].astype(float)
    V3_use = V3.copy()
    V3_use[2] += HOOP_Z_OFFSET_M
    V5 = w_map["V5"].astype(float)

    X0 = float(V5[0])          # plane X=const
    Z_hoop = float(V3[2])

    # Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6:
        fps = 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(str(OUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {OUT_VIDEO}")

    model = YOLO(MODEL_PATH)

    detections = []
    world_traj = []
    trail_px = []

    entered = False
    enter_streak = 0
    best_inside = None   # stores the "best" (most centered) inside instant

    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_sec = frame_id / fps

        results = model.predict(source=frame, verbose=False)
        det = choose_ball_detection_ultralytics(results[0], BALL_CLASS_ID)

        x1 = y1 = x2 = y2 = cx = cy = conf = np.nan
        Xw = None

        if det is not None:
            x1, y1, x2, y2, conf = det
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            if conf >= CONF_MIN:
                d_w = ray_world_from_pixel(cx, cy, K, dist, R_wc)
                Xw = intersect_ray_plane_X(C, d_w, X0)

        detections.append({
            "frame_id": frame_id, "t_sec": t_sec,
            "cx": cx, "cy": cy,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": conf if np.isfinite(conf) else np.nan
        })

        if Xw is not None and np.isfinite(conf) and conf >= CONF_MIN:
            # Distance to hoop center in world XY
            d_perp = float(np.hypot(Xw[0] - V3_use[0], Xw[1] - V3_use[1]))

            # "Fully passed" + "fits through the hoop"
            cond_z = (Xw[2] <= (Z_hoop - BALL_R_M))
            cond_r = (d_perp <= (HOOP_R_M - BALL_R_M + ENTER_MARGIN_M))

            if cond_z and cond_r:
                enter_streak += 1

                # Keep the best (most centered) inside frame
                score = d_perp
                if best_inside is None or score < best_inside["d_perp"]:
                    best_inside = {
                        "frame_id": frame_id,
                        "t_sec": t_sec,
                        "X": float(Xw[0]),
                        "Y": float(Xw[1]),
                        "Z": float(Xw[2]),
                        "d_perp": d_perp,
                        "enter_streak": enter_streak
                    }

                if enter_streak >= ENTER_N_CONSEC:
                    entered = True
            else:
                enter_streak = 0

                # Decision logic: descending crossing at Z=Z_hoop
                # (intentionally left as in your original script)

        # -------- overlay --------
        overlay = frame.copy()

        # bbox
        if np.isfinite(conf) and conf >= CONF_MIN:
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(overlay, (int(round(cx)), int(round(cy))), 3, (0, 255, 0), -1)

        # trail
        for i in range(1, len(trail_px)):
            cv2.line(overlay, trail_px[i - 1], trail_px[i], (0, 255, 255), 2)

        # hoop (projected polyline)
        draw_hoop_polyline(overlay, K, dist, rvec, tvec, V3_use)

        # label
        label = "ENTER" if entered else "MISS"
        col = (0, 255, 0) if entered else (0, 0, 255)
        cv2.putText(overlay, label, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 3, cv2.LINE_AA)

        writer.write(overlay)

        cv2.imshow("Overlay (ESC/q to quit)", overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

        frame_id += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    pd.DataFrame(detections).to_csv(OUT_DETECTIONS, index=False)
    pd.DataFrame(world_traj).to_csv(OUT_TRAJ_WORLD, index=False)

    decision = {
        "entered": bool(entered),
        "best_inside": best_inside,
        "hoop_center_world": V3.tolist(),
        "hoop_radius_m": HOOP_R_M,
        "tol_m": TOL_M,
        "pose_refine_rmse_px": rmse,
        "outputs": {
            "overlay_video": str(OUT_VIDEO),
            "detections_csv": str(OUT_DETECTIONS),
            "trajectory_world_csv": str(OUT_TRAJ_WORLD),
            "decision_json": str(OUT_DECISION),
        },
    }
    OUT_DECISION.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"[DONE] {OUT_VIDEO}")
    print(f"[DONE] {OUT_DECISION}")

    # Playback loop
    cap2 = cv2.VideoCapture(str(OUT_VIDEO))
    if cap2.isOpened():
        while True:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ok, fr = cap2.read()
                if not ok:
                    break
                cv2.imshow("Overlay loop (ESC/q to quit)", fr)
                key = cv2.waitKey(int(1000 / max(1.0, fps))) & 0xFF
                if key == 27 or key == ord("q"):
                    cap2.release()
                    cv2.destroyAllWindows()
                    return


if __name__ == "__main__":
    main()