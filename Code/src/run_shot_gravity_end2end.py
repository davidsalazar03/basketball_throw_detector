import json
from multiprocessing.util import debug
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parent

VIDEO_PATH = r"data/new_videos_raw/shot_4.mp4"
MODEL_PATH = r"runs/detect/train3/weights/best.pt"

CALIB_JSON = r"outputs/calibration_refined.json"
KEYPOINTS_WORLD_CSV = r"inputs/keypoints_world.csv"
KEYPOINTS_IMAGE_CSV = r"inputs/keypoints_image.csv"

OUT_DIR = Path(r"outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_VIDEO = OUT_DIR / "overlay_shot_1.mp4"
OUT_DETECTIONS = OUT_DIR / "ball_detections.csv"
OUT_TRAJ_WORLD = OUT_DIR / "ball_trajectory_world.csv"
OUT_DECISION = OUT_DIR / "shot_decision.json"


# ============================================================
# CONFIG
# ============================================================
SKIP_FIRST = 13
FIT_MAX_FRAC = 0.60

# Physics (Z up)
g = 9.81

# Hoop / ball
HOOP_DIAM_M = 0.45
HOOP_R_M = HOOP_DIAM_M / 2.0
HOOP_Z_OFFSET_M = -0.05
BALL_R_M = 0.12

# YOLO
CONF_MIN = 0.25
BALL_CLASS_ID = None

# Depth from bbox (sanity limits)
MIN_RPX = 4.0
MIN_Z_M = 1.0
MAX_Z_M = 30.0

# Depth scale calibration
K_SEARCH = np.linspace(0.7, 1.7, 51)
K_SCORE_BETA_Z = 0.6
K_SCORE_BETA_R = 1.0

# Base decision (3D)
ENTER_MARGIN_M = 0.02
REQUIRE_DESCENDING = True

# Ballistic fitting window (more permissive)
FIT_MAX_FRAC = 0.75

# Asymmetric vertical band (reduces false negatives)
Z_ABOVE_EPS = 0.25      # can be up to 25 cm above the hoop
Z_BELOW_BAND = 1.00     # can be up to 1.0 m below the hoop

# Streak (more permissive when there is 2D evidence)
ENTER_N_CONSEC = 2
ENTER_N_CONSEC_NO_2D = 4


VZ_MIN_DOWN = 0.5

# Ellipse gate (reinforcement)
ELLIPSE_MARGIN_PX = 18      # more permissive
USE_ELLIPSE_AS_SOFT = True  # does not hard-block

# Overlay
MAX_TRAIL = 140
SHOW_WINDOW = True


# ============================================================
# CALIB / POSE
# ============================================================
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

    obj = np.stack([w_map[k] for k in common], axis=0).astype(np.float64)
    img = np.stack([i_map[k] for k in common], axis=0).astype(np.float64)
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

    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - img_pts, axis=1)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return rvec, tvec, rmse


# ============================================================
# IMAGE -> normalized
# ============================================================
def undistort_to_normalized(u, v, K, dist):
    pts = np.array([[[u, v]]], dtype=np.float64)
    und = cv2.undistortPoints(pts, K, dist, P=None)
    x_n, y_n = und.reshape(2)
    return float(x_n), float(y_n)


# ============================================================
# YOLO helper
# ============================================================
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


# ============================================================
# HOOP overlay
# ============================================================
def draw_hoop_polyline(overlay, K, dist, rvec, tvec, hoop_center_world, hoop_r):
    thetas = np.linspace(0, 2*np.pi, 60, endpoint=False)
    pts = []
    for th in thetas:
        Pw = hoop_center_world + np.array([hoop_r*np.cos(th), hoop_r*np.sin(th), 0.0], dtype=float)
        uv, _ = cv2.projectPoints(Pw.reshape(1, 3), rvec, tvec, K, dist)
        u, v = uv.reshape(2)
        if np.isfinite(u) and np.isfinite(v):
            pts.append([u, v])
    if len(pts) >= 6:
        pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], True, (255, 0, 0), 2, cv2.LINE_AA)


# ============================================================
# 2D ellipse gate (soft)
# ============================================================
def point_in_hoop_ellipse(u, v, K, dist, rvec, tvec, hoop_center_world, hoop_r, margin_px=10):
    thetas = np.linspace(0, 2*np.pi, 80, endpoint=False)
    pts = []
    for th in thetas:
        Pw = hoop_center_world + np.array([hoop_r*np.cos(th), hoop_r*np.sin(th), 0.0], dtype=float)
        uv, _ = cv2.projectPoints(Pw.reshape(1, 3), rvec, tvec, K, dist)
        uu, vv = uv.reshape(2)
        if np.isfinite(uu) and np.isfinite(vv):
            pts.append([uu, vv])

    if len(pts) < 20:
        return False

    pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    ell = cv2.fitEllipse(pts)
    (cx, cy), (MA, ma), angle_deg = ell

    ang = np.deg2rad(angle_deg)
    c, s = np.cos(ang), np.sin(ang)

    dx = float(u - cx)
    dy = float(v - cy)

    xr = c * dx + s * dy
    yr = -s * dx + c * dy

    a = 0.5 * float(MA) + float(margin_px)
    b = 0.5 * float(ma) + float(margin_px)
    if a <= 1e-6 or b <= 1e-6:
        return False

    val = (xr * xr) / (a * a) + (yr * yr) / (b * b)
    return val <= 1.0


# ============================================================
# 3D from bbox size with depth scale k
# ============================================================
def Zc_from_bbox(K, x1, y1, x2, y2, ball_r_m, k_depth):
    w = float(x2 - x1)
    h = float(y2 - y1)
    r_px = 0.25 * (w + h)
    if not np.isfinite(r_px) or r_px < MIN_RPX:
        return None

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    f = 0.5 * (fx + fy)

    Zc = k_depth * (f * ball_r_m / r_px)
    Zc = float(np.clip(Zc, MIN_Z_M, MAX_Z_M))
    return Zc


def world_point_from_det(cx, cy, Zc, K, dist, R_wc, C_world):
    x_n, y_n = undistort_to_normalized(cx, cy, K, dist)
    Xc = np.array([x_n * Zc, y_n * Zc, Zc], dtype=float)
    Xw = (R_wc @ Xc.reshape(3, 1)).reshape(3) + C_world.reshape(3)
    return Xw


# ============================================================
# Ballistic LS
# ============================================================
def fit_ballistic_ls(times, Xw):
    a = np.array([0.0, 0.0, -g], dtype=float)

    t = times.reshape(-1)
    A = np.stack([np.ones_like(t), t], axis=1)

    Y = Xw - 0.5 * (t[:, None] ** 2) * a[None, :]
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)

    p0 = coef[0, :]
    v0 = coef[1, :]

    Xhat = p0[None, :] + v0[None, :] * t[:, None] + 0.5 * (t[:, None] ** 2) * a[None, :]
    resid = np.sqrt(np.sum((Xw - Xhat) ** 2, axis=1))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return p0, v0, rmse


def p_of_t(p0, v0, tt):
    a = np.array([0.0, 0.0, -g], dtype=float)
    tt = np.asarray(tt, dtype=float)
    return p0[None, :] + v0[None, :] * tt[:, None] + 0.5 * (tt[:, None] ** 2) * a[None, :]


def v_of_t(v0, tt):
    a = np.array([0.0, 0.0, -g], dtype=float)
    tt = np.asarray(tt, dtype=float)
    return v0[None, :] + tt[:, None] * a[None, :]


# ============================================================
# Choose best k_depth
# ============================================================
def choose_k_depth(times, det_rows, hoop_center, Z_hoop, K, dist, R_wc, C_world):
    best = None

    for k_depth in K_SEARCH:
        Xw_list = []
        for r in det_rows:
            Zc = Zc_from_bbox(K, r["x1"], r["y1"], r["x2"], r["y2"], BALL_R_M, k_depth)
            if Zc is None:
                Xw_list.append(None)
                continue
            Xw = world_point_from_det(r["cx"], r["cy"], Zc, K, dist, R_wc, C_world)
            Xw_list.append(Xw)

        mask = np.array([x is not None for x in Xw_list], dtype=bool)
        if mask.sum() < 6:
            continue

        Xw = np.stack([Xw_list[i] for i in range(len(Xw_list)) if mask[i]], axis=0)
        tt = times[mask]

        p0, v0, rmse = fit_ballistic_ls(tt, Xw)

        P = p_of_t(p0, v0, tt)
        d = np.hypot(P[:, 0] - hoop_center[0], P[:, 1] - hoop_center[1])

        i_min = int(np.argmin(d))
        d_min = float(d[i_min])
        z_at = float(P[i_min, 2])

        score = (K_SCORE_BETA_R * d_min) + (K_SCORE_BETA_Z * abs(z_at - Z_hoop)) + 0.15 * rmse

        if best is None or score < best["score"]:
            best = {
                "k_depth": float(k_depth),
                "score": float(score),
                "rmse": float(rmse),
                "d_min": d_min,
                "z_at": z_at,
                "p0": p0,
                "v0": v0,
            }

    return best


# ============================================================
# MAIN
# ============================================================
def main():
    from ultralytics import YOLO

    entered = False
    enter_frame = None
    best_inside = None

    calib = load_calibration(str(CALIB_JSON))
    common, obj_pts, img_pts, w_map = load_keypoints(str(KEYPOINTS_WORLD_CSV), str(KEYPOINTS_IMAGE_CSV))

    rvec, tvec, rmse_px = refine_pose_solvepnp(calib, obj_pts, img_pts)
    print(f"[POSE] solvePnP RMSE (px): {rmse_px:.3f} | pts={common}")

    K = calib["K"]
    dist = calib["dist"]

    R_cw, _ = cv2.Rodrigues(rvec)
    R_wc = R_cw.T
    C_world = (-R_wc @ tvec.reshape(3, 1)).reshape(3)

    V3 = w_map["V3"].astype(float)
    hoop_center = V3.copy()
    hoop_center[2] += HOOP_Z_OFFSET_M
    Z_hoop = float(hoop_center[2])

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6:
        fps = 30.0

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fit_last_frame = int(round(FIT_MAX_FRAC * (total_frames - 1)))

    writer = cv2.VideoWriter(
        str(OUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {OUT_VIDEO}")

    model = YOLO(str(MODEL_PATH))

    detections = []
    frames_cache = []
    centers_by_frame = {}

    fit_rows = []
    fit_times = []

    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frames_cache.append(frame.copy())
        t_sec = frame_id / fps

        cx = cy = x1 = y1 = x2 = y2 = conf = np.nan

        if frame_id >= SKIP_FIRST:
            results = model.predict(source=frame, verbose=False)
            det = choose_ball_detection_ultralytics(results[0], BALL_CLASS_ID)
            if det is not None:
                x1, y1, x2, y2, conf = det
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)

        detections.append({
            "frame_id": frame_id, "t_sec": t_sec,
            "cx": cx, "cy": cy,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": conf if np.isfinite(conf) else np.nan
        })

        if np.isfinite(conf) and conf >= CONF_MIN:
            centers_by_frame[frame_id] = (float(cx), float(cy))
            if frame_id <= fit_last_frame:
                fit_rows.append({
                    "cx": float(cx), "cy": float(cy),
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2)
                })
                fit_times.append(float(t_sec))

        frame_id += 1

    cap.release()

    world_traj = []
    fit_rmse_m = None

    debug = {
        "fit_last_frame": int(fit_last_frame),
        "fit_used_pts": int(len(fit_times)),
        "k_depth_best": None,
        "k_score": None,
        "event_enter_frame": None,
        "event_t_sec": None,
        "rule": None,
        "r_thresh_used": float(HOOP_R_M + BALL_R_M + ENTER_MARGIN_M),
        "enter_n_consec": int(ENTER_N_CONSEC),
        "enter_n_consec_no_2d": int(ENTER_N_CONSEC_NO_2D),
        "vz_min_down": float(VZ_MIN_DOWN),
        "ellipse_margin_px": int(ELLIPSE_MARGIN_PX),
        "pose_rmse_px": float(rmse_px),
    }

    if len(fit_times) >= 6:
        fit_times = np.array(fit_times, dtype=float)

        best_k = choose_k_depth(
            times=fit_times,
            det_rows=fit_rows,
            hoop_center=hoop_center,
            Z_hoop=Z_hoop,
            K=K, dist=dist, R_wc=R_wc, C_world=C_world
        )

        if best_k is None:
            print("[WARN] Could not estimate k_depth (too few valid points).")
        else:
            k_depth = best_k["k_depth"]
            p0 = best_k["p0"]
            v0 = best_k["v0"]
            fit_rmse_m = best_k["rmse"]

            debug["k_depth_best"] = float(k_depth)
            debug["k_score"] = float(best_k["score"])

            print(f"[DEPTH] Best k_depth: {k_depth:.3f} | score={best_k['score']:.4f} | rmse={fit_rmse_m:.4f}")

            all_times = np.arange(len(frames_cache), dtype=float) / fps
            P_all = p_of_t(p0, v0, all_times)
            V_all = v_of_t(v0, all_times)

            for k in range(len(frames_cache)):
                if k < SKIP_FIRST:
                    world_traj.append({"frame_id": k, "t_sec": float(all_times[k]), "X": np.nan, "Y": np.nan, "Z": np.nan})
                else:
                    world_traj.append({
                        "frame_id": k,
                        "t_sec": float(all_times[k]),
                        "X": float(P_all[k, 0]),
                        "Y": float(P_all[k, 1]),
                        "Z": float(P_all[k, 2]),
                    })

            r_thresh = float(HOOP_R_M + BALL_R_M + ENTER_MARGIN_M)
            k_max = min(int(fit_last_frame), len(P_all) - 1)

            streak = 0
            for k in range(SKIP_FIRST + 1, k_max + 1):
                vz = float(V_all[k, 2])
                if REQUIRE_DESCENDING and vz >= -VZ_MIN_DOWN:
                    streak = 0
                    continue

                d_xy = float(np.hypot(P_all[k, 0] - hoop_center[0], P_all[k, 1] - hoop_center[1]))
                dz = float(abs(P_all[k, 2] - Z_hoop))

                Pz = float(P_all[k, 2])
                cond_z_band = (Pz <= (Z_hoop + Z_ABOVE_EPS)) and (Pz >= (Z_hoop - Z_BELOW_BAND))
                cond3d = (d_xy <= r_thresh) and cond_z_band

                inside = False
                if k in centers_by_frame:
                    u, v = centers_by_frame[k]
                    inside = point_in_hoop_ellipse(
                        u, v, K, dist, rvec, tvec, hoop_center, HOOP_R_M, margin_px=ELLIPSE_MARGIN_PX
                    )

                if not cond3d:
                    streak = 0
                    continue

                # >>> THIS IS THE DIFFERENCE:
                # - if inside ellipse: use the normal streak requirement
                # - if not inside: still allowed, but requires a longer streak
                streak += 1
                needed = ENTER_N_CONSEC if (inside or not USE_ELLIPSE_AS_SOFT) else ENTER_N_CONSEC_NO_2D

                if streak >= needed:
                    entered = True
                    enter_frame = int(k)
                    best_inside = {
                        "frame_id": int(k),
                        "t_sec": float(all_times[k]),
                        "X": float(P_all[k, 0]),
                        "Y": float(P_all[k, 1]),
                        "Z": float(P_all[k, 2]),
                        "d_perp": d_xy,
                        "dz": dz,
                        "vz": vz,
                        "inside_ellipse": bool(inside),
                        "rule": "3D_band + ellipse_soft + streak",
                        "streak": int(streak),
                        "needed": int(needed),
                    }
                    break

            debug["event_enter_frame"] = enter_frame
            debug["z_above_eps"] = float(Z_ABOVE_EPS)
            debug["z_below_band"] = float(Z_BELOW_BAND)

            if enter_frame is not None:
                debug["event_t_sec"] = float(all_times[enter_frame])
                debug["rule"] = str(best_inside.get("rule", None))

    # ---------------- OVERLAY ----------------
    trail_px = []
    for k, frame in enumerate(frames_cache):
        overlay = frame.copy()
        draw_hoop_polyline(overlay, K, dist, rvec, tvec, hoop_center, HOOP_R_M)

        det_row = detections[k]
        conf = det_row["conf"]
        if np.isfinite(conf) and conf >= CONF_MIN:
            x1, y1, x2, y2 = det_row["x1"], det_row["y1"], det_row["x2"], det_row["y2"]
            cx, cy = det_row["cx"], det_row["cy"]
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(overlay, (int(round(cx)), int(round(cy))), 3, (0, 255, 0), -1)

            inside = point_in_hoop_ellipse(cx, cy, K, dist, rvec, tvec, hoop_center, HOOP_R_M, margin_px=ELLIPSE_MARGIN_PX)
            if inside:
                cv2.putText(overlay, "IN-ELLIPSE", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)

        pt = None
        if k in centers_by_frame:
            u, v = centers_by_frame[k]
            pt = (int(round(u)), int(round(v)))

        if pt is not None:
            trail_px.append(pt)
            trail_px = trail_px[-MAX_TRAIL:]
            cv2.circle(overlay, pt, 5, (0, 255, 255), -1)
            for i in range(1, len(trail_px)):
                cv2.line(overlay, trail_px[i - 1], trail_px[i], (0, 255, 255), 2)

        is_enter_now = (enter_frame is not None and k >= enter_frame)
        label = "ENTER" if is_enter_now else "MISS"
        col = (0, 255, 0) if is_enter_now else (0, 0, 255)
        cv2.putText(overlay, label, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 3, cv2.LINE_AA)

        writer.write(overlay)

        if SHOW_WINDOW:
            cv2.imshow("Overlay (ESC/q to quit)", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    writer.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    pd.DataFrame(detections).to_csv(OUT_DETECTIONS, index=False)
    pd.DataFrame(world_traj).to_csv(OUT_TRAJ_WORLD, index=False)

    decision = {
        "entered": bool(entered),
        "enter_frame": enter_frame,
        "best_inside": best_inside,
        "hoop_center_world": hoop_center.tolist(),
        "hoop_radius_m": HOOP_R_M,
        "ball_radius_m": BALL_R_M,
        "skip_first": int(SKIP_FIRST),
        "fit_max_frac": float(FIT_MAX_FRAC),
        "pose_refine_rmse_px": float(rmse_px),
        "ballistic_fit_rmse_m": float(fit_rmse_m) if fit_rmse_m is not None else None,
        "debug": debug,
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


if __name__ == "__main__":
    main()
