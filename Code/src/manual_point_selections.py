# label_court_points.py (UPDATED)
#
# What it does now (summary):
# - Keeps the manual selection wizard (points -> lines -> compute VPs/intersections -> hoop -> conic)
# - At startup, if 4 files exist in outputs/ (points, lines, VPs, conic), it asks whether you want to reuse them:
#     - N/n: reuse (view-only), draw everything and do not generate new outputs
#     - Y/y: redo manual selection from scratch
# - On SAVE (S), it saves outputs and also automatically generates in inputs/:
#     - inputs/image_points.csv (id,u,v)  [compatible with scripts 2/3/4]
#     - inputs/vp_vertical.txt (from VP_B)
#     - inputs/keypoints_image.csv (V1,V3,V4,V5)
#     - inputs/keypoints_world.csv (V1,V3,V4,V5) read from inputs/world_points_coord.csv
#
# Requirements: opencv-python, numpy

from pathlib import Path
import os
import cv2
import csv
import numpy as np

# ---------------- CONFIG ----------------
IMAGE_PATH = r"data/new_dataset/images/shot_1_f0000.jpg"

# outputs (wizard artifacts)
OUT_POINTS_CSV = r"outputs/court_points_image.csv"
OUT_LINES_CSV  = r"outputs/court_lines_image.csv"
OUT_VP_CSV     = r"outputs/vanishing_points.csv"
OUT_CONIC_CSV  = r"outputs/hoop_conic.csv"

# inputs (artifacts for the rest of the pipeline)
INPUTS_DIR = Path("inputs")
INPUT_IMAGE_POINTS_CSV = INPUTS_DIR / "image_points.csv"
INPUT_VP_VERTICAL_TXT  = INPUTS_DIR / "vp_vertical.txt"
INPUT_KEYPOINTS_IMAGE  = INPUTS_DIR / "keypoints_image.csv"
INPUT_KEYPOINTS_WORLD  = INPUTS_DIR / "keypoints_world.csv"
INPUT_WORLD_POINTS_CSV = INPUTS_DIR / "world_points_coord.csv"

WINDOW_NAME = "Court Wizard (ESC=exit | U=undo | S=save | ENTER=next/compute)"
RADIUS = 4

# Step 1: points clicked directly
DIRECT_POINTS = ["A1", "A2", "A3", "A4", "V1", "V2"]

# Step 2: manually drawn lines (2 clicks each)
MANUAL_LINES = [
    "LP1", "LP2", "LP3",
    "LQ1", "LQ2", "LQ3",
    "LB1", "LB2", "LB3", "LB4",
]

# Parallel families (for VP via SVD)
LINE_FAMILIES = {
    "VP_P": ["LP1", "LP2", "LP3"],
    "VP_Q": ["LQ1", "LQ2", "LQ3"],   # LQ4 will be derived
    "VP_B": ["LB1", "LB2", "LB3", "LB4"],  # considered "vertical" (as you said)
}

# Step 4: hoop (conic fit)
HOOP_MIN_PTS = 5
HOOP_TARGET_PTS = 8  # you can click more than this; ENTER finishes

# --------- COLORS PER FAMILY (OpenCV BGR) ----------
COLOR_LP = (0, 0, 255)        # red
COLOR_LQ = (170, 255, 170)    # light green
COLOR_LB = (255, 200, 120)    # light blue
COLOR_DERIVED = (180, 0, 255) # purple fallback

COLOR_POINTS_MANUAL = (0, 255, 0)
COLOR_POINTS_DERIVED = (255, 255, 0)
COLOR_HOOP_PTS = (0, 255, 255)  # yellow
COLOR_V3 = (0, 140, 255)        # orange


def line_family_color(line_id: str):
    if line_id.startswith("LP"):
        return COLOR_LP
    if line_id.startswith("LQ"):
        return COLOR_LQ
    if line_id.startswith("LB"):
        return COLOR_LB
    return (0, 165, 255)


# --------------- STATE -------------------
state = {
    "stage": "points",     # "points" -> "lines" -> "hoop" -> "done"
    "pts": {},             # id -> (u,v) in pixels
    "lines": {},           # id -> {"p1":(x,y), "p2":(x,y), "h":(a,b,c)}
    "derived_lines": {},   # id -> homog line (a,b,c)
    "derived_points": {},  # id -> (u,v)
    "vps": {},             # id -> (vx,vy,vw)

    "i_point": 0,
    "i_line": 0,
    "pending_p1": None,    # for 2-click line drawing

    "hoop_pts": [],        # list of hoop points (u,v)
    "conic": None,         # (a,b,c,d,e,f) normalized
    "conic_center": None,  # (u,v)

    "img": None,
    "canvas": None,

    # startup/reuse
    "startup_mode": "prompt",   # "prompt" | "manual" | "reuse"
    "view_only": False,         # True when loading existing files
}


# =============== GEOMETRY (H) ===============

def to_h_point(p):
    return np.array([float(p[0]), float(p[1]), 1.0], dtype=float)

def normalize_line(l):
    l = np.array(l, dtype=float)
    n = np.hypot(l[0], l[1])
    if n > 1e-12:
        l = l / n
    return l

def segment_to_line_h(p1, p2):
    l = np.cross(to_h_point(p1), to_h_point(p2))  # (a,b,c)
    return tuple(normalize_line(l).tolist())

def intersect_lines_h(l1, l2):
    p = np.cross(np.array(l1, dtype=float), np.array(l2, dtype=float))  # (x,y,w)
    if abs(p[2]) < 1e-12:
        return None
    return (float(p[0] / p[2]), float(p[1] / p[2]))

def line_through_point_and_vp(point_uv, vp_h):
    p = to_h_point(point_uv)
    vp = np.array(vp_h, dtype=float)
    l = np.cross(p, vp)
    return tuple(normalize_line(l).tolist())

def estimate_vanishing_point_svd(lines_h):
    """
    lines_h: list of (a,b,c). Solve A v = 0 via SVD.
    """
    if len(lines_h) < 2:
        raise ValueError("You need at least 2 lines to estimate a VP.")
    A = np.array(lines_h, dtype=float)
    _, _, Vt = np.linalg.svd(A)
    vp = Vt[-1, :]  # smallest singular value
    if abs(vp[2]) > 1e-12:
        vp = vp / vp[2]
    else:
        n = np.hypot(vp[0], vp[1])
        if n > 1e-12:
            vp = vp / n
    return tuple(vp.tolist())


# =============== CONIC FIT (hoop) ===============

def fit_conic_fitzgibbon(points_uv):
    """
    Least-squares conic fitting (Fitzgibbon et al.) in general form:
      ax^2 + bxy + cy^2 + dx + ey + f = 0

    Returns (a,b,c,d,e,f) with a simple normalization.
    """
    pts = np.asarray(points_uv, dtype=float)
    if pts.shape[0] < HOOP_MIN_PTS:
        raise ValueError(f"You need at least {HOOP_MIN_PTS} points for conic fitting.")

    x = pts[:, 0]
    y = pts[:, 1]
    D = np.column_stack([x*x, x*y, y*y, x, y, np.ones_like(x)])  # Nx6

    _, _, Vt = np.linalg.svd(D)
    p = Vt[-1, :]  # (a,b,c,d,e,f)

    n = np.linalg.norm(p)
    if n > 1e-12:
        p = p / n

    return tuple(p.tolist())

def conic_center_from_params(p):
    """
    For a general conic:
      ax^2 + bxy + cy^2 + dx + ey + f = 0
    The center solves:
      [2a  b] [x] = [-d]
      [ b 2c] [y]   [-e]
    """
    a, b, c, d, e, f0 = map(float, p)
    M = np.array([[2*a, b],
                  [b, 2*c]], dtype=float)
    rhs = np.array([-d, -e], dtype=float)

    det = np.linalg.det(M)
    if abs(det) < 1e-12:
        return None

    xy = np.linalg.solve(M, rhs)
    return (float(xy[0]), float(xy[1]))

def is_ellipse_like(p):
    a, b, c, d, e, f0 = map(float, p)
    return (b*b - 4*a*c) < 0

def draw_conic(canvas, p, color=(255, 255, 255), thickness=2, step=2):
    a, b, c, d, e, f0 = map(float, p)
    H, W = canvas.shape[:2]

    pts_draw = []
    for x in range(0, W, step):
        A = c
        B = b*x + e
        C = a*(x*x) + d*x + f0

        if abs(A) < 1e-12:
            if abs(B) > 1e-12:
                y = -C / B
                if 0 <= y < H:
                    pts_draw.append((x, int(y)))
            continue

        disc = B*B - 4*A*C
        if disc < 0:
            continue
        s = np.sqrt(disc)
        y1 = (-B + s) / (2*A)
        y2 = (-B - s) / (2*A)

        if 0 <= y1 < H:
            pts_draw.append((x, int(y1)))
        if 0 <= y2 < H:
            pts_draw.append((x, int(y2)))

    for i in range(1, len(pts_draw)):
        cv2.line(canvas, pts_draw[i-1], pts_draw[i], color, thickness)


# =============== DRAW HELPERS ===============

def draw_infinite_line(img, l, color=(255, 0, 255), thickness=2):
    a, b, c = l
    h, w = img.shape[:2]
    pts = []
    for x in [0, w - 1]:
        if abs(b) > 1e-12:
            y = -(a * x + c) / b
            if 0 <= y <= h - 1:
                pts.append((int(x), int(y)))
    for y in [0, h - 1]:
        if abs(a) > 1e-12:
            x = -(b * y + c) / a
            if 0 <= x <= w - 1:
                pts.append((int(x), int(y)))
    if len(pts) >= 2:
        cv2.line(img, pts[0], pts[1], color, thickness)

def _rect_intersects_any(rect, rects, pad=2):
    x, y, w, h = rect
    for (rx, ry, rw, rh) in rects:
        if (x - pad) < (rx + rw) and (x + w + pad) > rx and (y - pad) < (ry + rh) and (y + h + pad) > ry:
            return True
    return False

def _point_inside_rect(pt, rect, pad=2):
    x, y, w, h = rect
    px, py = pt
    return (x - pad) <= px <= (x + w + pad) and (y - pad) <= py <= (y + h + pad)

def place_label_no_overlap(canvas, text, anchor_xy, placed_label_rects, avoid_points, color, font_scale=0.6, thickness=2):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    th = th + baseline

    ax, ay = anchor_xy
    candidates = [
        (ax + 8, ay - 8),
        (ax + 8, ay + th + 8),
        (ax - tw - 8, ay - 8),
        (ax - tw - 8, ay + th + 8),
        (ax + 8, ay - th - 8),
        (ax - tw - 8, ay - th - 8),
    ]

    H, W = canvas.shape[:2]

    best_pos = candidates[0]
    best_rect = (0, 0, tw, th)
    best_score = 10**9

    for (x, y) in candidates:
        x = int(np.clip(x, 0, W - tw - 1))
        y = int(np.clip(y, th + 1, H - 1))
        rect = (x, y - th, tw, th)

        score = 0
        if _rect_intersects_any(rect, placed_label_rects, pad=3):
            score += 1000
        for p in avoid_points:
            if _point_inside_rect(p, rect, pad=3):
                score += 200
        score += abs((x - ax)) + abs((y - ay))

        if score < best_score:
            best_score = score
            best_pos = (x, y)
            best_rect = rect

    x, y = best_pos
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    placed_label_rects.append(best_rect)


def redraw():
    canvas = state["img"].copy()
    H, W = canvas.shape[:2]

    placed_label_rects = []
    avoid_points = []

    # 1) manual points
    for pid, (u, v) in state["pts"].items():
        cv2.circle(canvas, (int(u), int(v)), RADIUS, COLOR_POINTS_MANUAL, -1)
        place_label_no_overlap(
            canvas, pid, (int(u), int(v)),
            placed_label_rects, avoid_points, COLOR_POINTS_MANUAL
        )
        avoid_points.append((int(u), int(v)))

    # 1b) derived points
    for pid, (u, v) in state["derived_points"].items():
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(canvas, (int(u), int(v)), RADIUS, COLOR_POINTS_DERIVED, -1)
            place_label_no_overlap(
                canvas, pid, (int(u), int(v)),
                placed_label_rects, avoid_points, COLOR_POINTS_DERIVED
            )
            avoid_points.append((int(u), int(v)))

    # 1c) hoop points
    for (u, v) in state["hoop_pts"]:
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(canvas, (int(u), int(v)), RADIUS, COLOR_HOOP_PTS, -1)
            avoid_points.append((int(u), int(v)))

    # 1d) V3 (hoop center)
    if state["conic_center"] is not None:
        u, v = state["conic_center"]
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(canvas, (int(u), int(v)), 6, COLOR_V3, -1)
            place_label_no_overlap(canvas, "V3", (int(u), int(v)), placed_label_rects, avoid_points, COLOR_V3, font_scale=0.7)

    # 2) manual lines (segment + infinite), with family color
    for lid, obj in state["lines"].items():
        (x1, y1) = obj["p1"]
        (x2, y2) = obj["p2"]
        col = line_family_color(lid)

        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)

        ax = int(min(x1, x2))
        ay = int(min(y1, y2))
        place_label_no_overlap(canvas, lid, (ax, ay), placed_label_rects, avoid_points, col)

        draw_infinite_line(canvas, obj["h"], color=col, thickness=2)

    # 3) derived lines (infinite only)
    for lid, l in state["derived_lines"].items():
        col = line_family_color(lid) if (lid.startswith("LP") or lid.startswith("LQ") or lid.startswith("LB")) else COLOR_DERIVED
        draw_infinite_line(canvas, l, color=col, thickness=2)

        idx = list(state["derived_lines"].keys()).index(lid)
        place_label_no_overlap(canvas, lid, (20, 160 + 22 * idx), placed_label_rects, avoid_points, col)

    # 4) draw conic (debug)
    if state["conic"] is not None:
        draw_conic(canvas, state["conic"], color=(255, 255, 255), thickness=2, step=2)

    # HUD
    cv2.rectangle(canvas, (10, 10), (1500, 95), (0, 0, 0), -1)

    if state.get("startup_mode") == "prompt":
        msg = "START | Existing outputs found. Press Y (re-do) or N (reuse/view-only)."
    elif state.get("view_only", False):
        msg = "VIEW-ONLY | Reused existing outputs. ESC to exit."
    elif state["stage"] == "points":
        next_id = DIRECT_POINTS[state["i_point"]] if state["i_point"] < len(DIRECT_POINTS) else None
        msg = f"STEP 1/4 | Select point: {next_id} ({state['i_point']+1}/{len(DIRECT_POINTS)})" if next_id else \
              "STEP 1/4 | Points done. Now draw lines (STEP 2)."
    elif state["stage"] == "lines":
        next_l = MANUAL_LINES[state["i_line"]] if state["i_line"] < len(MANUAL_LINES) else None
        if next_l:
            if state["pending_p1"] is None:
                msg = f"STEP 2/4 | Draw line {next_l}: click P1"
            else:
                msg = f"STEP 2/4 | Draw line {next_l}: click P2"
        else:
            msg = "STEP 2/4 | Lines done. Press ENTER to compute points/VPs (STEP 3)."
    elif state["stage"] == "hoop":
        n = len(state["hoop_pts"])
        extra = " (min OK)" if n >= HOOP_MIN_PTS else " (need more)"
        msg = f"STEP 4/4 | Click hoop points (n={n}){extra}. ENTER=fit conic, U=undo last hoop point."
    else:
        msg = "DONE | Press S to save all (outputs + inputs/*). U=undo goes back one step."

    cv2.putText(canvas, msg, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(canvas, "ESC=exit | U=undo | S=save | ENTER=next/compute/fit", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # STARTUP PROMPT OVERLAY
    if state.get("startup_mode") == "prompt":
        overlay = canvas.copy()
        cv2.rectangle(overlay, (10, 110), (1500, 260), (0, 0, 0), -1)
        alpha = 0.7
        canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

        msg1 = "Existing annotations found in outputs/ (points, lines, vanishing points, conic)."
        msg2 = "Do you wish to proceed with a new manual selection?"
        msg3 = "[Y] re-do selection from scratch  |  [N] reuse existing (view-only)"
        cv2.putText(canvas, msg1, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(canvas, msg2, (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (180, 255, 180), 2)
        cv2.putText(canvas, msg3, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (180, 255, 180), 2)

    state["canvas"] = canvas


# =============== COMPUTE (RULES) ===============

def get_line(lid):
    if lid in state["lines"]:
        return state["lines"][lid]["h"]
    if lid in state["derived_lines"]:
        return state["derived_lines"][lid]
    raise KeyError(f"Missing line {lid}")

def get_point(pid):
    if pid in state["pts"]:
        return state["pts"][pid]
    if pid in state["derived_points"]:
        return state["derived_points"][pid]
    raise KeyError(f"Missing point {pid}")

def compute_intersections_direct():
    rules = {
        "O":  ("LP1", "LQ1"),
        "C1": ("LP1", "LQ2"),
        "C2": ("LP1", "LQ3"),
        "C3": ("LQ2", "LP3"),
        "C4": ("LQ3", "LP3"),

        "B2": ("LP2", "LB1"),
        "B3": ("LP2", "LB2"),
        "B4": ("LP2", "LB3"),
        "B5": ("LP2", "LB4"),
    }
    for pid, (l1, l2) in rules.items():
        p = intersect_lines_h(get_line(l1), get_line(l2))
        if p is not None:
            state["derived_points"][pid] = p

def compute_vanishing_points():
    for vp_id, line_ids in LINE_FAMILIES.items():
        lines_h = [get_line(lid) for lid in line_ids]
        state["vps"][vp_id] = estimate_vanishing_point_svd(lines_h)

def compute_projected_lines_and_points():
    VP_Q = state["vps"]["VP_Q"]
    VP_P = state["vps"]["VP_P"]
    VP_B = state["vps"]["VP_B"]

    # LQ4 = (LQ through V1)
    state["derived_lines"]["LQ4"] = line_through_point_and_vp(get_point("V1"), VP_Q)

    projected_point_rules = {
        "B1": (("LQ", "O"), "LP2"),

        "P1": (("LQ", "A1"), "LP1"),
        "P2": (("LQ", "V1"), "LP1"),
        "P3": (("LQ", "A2"), "LP1"),

        "Q1": (("LP", "A1"), "LQ2"),
        "Q2": (("LP", "A2"), "LQ3"),

        "V5": (("LQ", "V1"), "LP3"),
    }

    def proj_line(fam, pid):
        if fam == "LQ":
            return line_through_point_and_vp(get_point(pid), VP_Q)
        if fam == "LP":
            return line_through_point_and_vp(get_point(pid), VP_P)
        if fam == "LB":
            return line_through_point_and_vp(get_point(pid), VP_B)
        raise ValueError(f"Unknown family: {fam}")

    for out_pid, ((fam, through_pid), with_line) in projected_point_rules.items():
        l_proj = proj_line(fam, through_pid)
        p = intersect_lines_h(l_proj, get_line(with_line))
        if p is not None:
            state["derived_points"][out_pid] = p

def compute_all():
    state["derived_points"] = {}
    state["derived_lines"] = {}
    state["vps"] = {}

    compute_intersections_direct()
    compute_vanishing_points()
    compute_projected_lines_and_points()


# =============== CONIC PIPELINE (STEP 4) ===============

def fit_hoop_conic_and_center():
    p = fit_conic_fitzgibbon(state["hoop_pts"])
    center = conic_center_from_params(p)

    state["conic"] = p
    state["conic_center"] = center

    if center is not None:
        state["derived_points"]["V3"] = center

        # compute V4 as intersection of: (line through center and VP_B) with LQ4
        if "VP_B" in state.get("vps", {}) and ("LQ4" in state.get("derived_lines", {})) or ("LQ4" in state.get("lines", {})):
            VP_B = state["vps"]["VP_B"]
            lb_on_v3 = line_through_point_and_vp(center, VP_B)
            p_v4 = intersect_lines_h(lb_on_v3, get_line("LQ4"))
            if p_v4 is not None:
                state["derived_points"]["V4"] = p_v4


# =============== IO (outputs) ===============

def save_points_csv(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    all_ids = sorted(set(list(state["pts"].keys()) + list(state["derived_points"].keys())))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "u", "v", "source"])
        for pid in all_ids:
            if pid in state["pts"]:
                u, v = state["pts"][pid]
                w.writerow([pid, f"{u:.2f}", f"{v:.2f}", "manual_point"])
            else:
                u, v = state["derived_points"][pid]
                w.writerow([pid, f"{u:.2f}", f"{v:.2f}", "computed"])
    print(f"[DONE] Points CSV: {path}")

def save_lines_csv(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["line_id", "type", "x1", "y1", "x2", "y2", "a", "b", "c"])
        for lid, obj in state["lines"].items():
            a, b, c = obj["h"]
            x1, y1 = obj["p1"]
            x2, y2 = obj["p2"]
            w.writerow([lid, "manual", int(x1), int(y1), int(x2), int(y2),
                        f"{a:.8f}", f"{b:.8f}", f"{c:.8f}"])
        for lid, l in state["derived_lines"].items():
            a, b, c = l
            w.writerow([lid, "derived", "", "", "", "", f"{a:.8f}", f"{b:.8f}", f"{c:.8f}"])
    print(f"[DONE] Lines CSV: {path}")

def save_vps_csv(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["vp_id", "vx", "vy", "vw"])
        for vp_id, vp in state["vps"].items():
            vx, vy, vw = vp
            w.writerow([vp_id, f"{vx:.8f}", f"{vy:.8f}", f"{vw:.8f}"])
    print(f"[DONE] VP CSV: {path}")

def save_conic_csv(path):
    if state["conic"] is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    a, b, c, d, e, f0 = state["conic"]
    cx, cy = ("", "")
    if state["conic_center"] is not None:
        cx, cy = state["conic_center"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "c", "d", "e", "f", "center_u", "center_v", "ellipse_like"])
        w.writerow([f"{a:.12e}", f"{b:.12e}", f"{c:.12e}", f"{d:.12e}", f"{e:.12e}", f"{f0:.12e}",
                    f"{cx:.6f}" if cx != "" else "", f"{cy:.6f}" if cy != "" else "",
                    int(is_ellipse_like(state["conic"]))])
    print(f"[DONE] Conic CSV: {path}")


# =============== IO (inputs) ===============

def export_inputs_image_points(dst_path: Path):
    """
    inputs/image_points.csv expected by scripts 2/3/4:
      id,u,v
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    all_ids = sorted(set(list(state["pts"].keys()) + list(state["derived_points"].keys())))
    with dst_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "u", "v"])
        for pid in all_ids:
            if pid in state["pts"]:
                u, v = state["pts"][pid]
            else:
                u, v = state["derived_points"][pid]
            w.writerow([pid, f"{u:.6f}", f"{v:.6f}"])
    print(f"[DONE] inputs/image_points.csv: {dst_path}")

def export_inputs_vp_vertical(dst_path: Path):
    """
    inputs/vp_vertical.txt expected by scripts 3/4.
    Uses VP_B (vertical).
    Format: "u v w" (single line)
    """
    if "VP_B" not in state["vps"]:
        raise RuntimeError("VP_B is not available in state['vps']. Press ENTER (compute) before saving.")
    vx, vy, vw = state["vps"]["VP_B"]
    # normalize to w=1 when possible
    if abs(vw) > 1e-12:
        vx, vy, vw = vx / vw, vy / vw, 1.0
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(f"{vx:.12g} {vy:.12g} {vw:.12g}\n", encoding="utf-8")
    print(f"[DONE] inputs/vp_vertical.txt: {dst_path}")

def read_world_points_map(path: Path):
    """
    Read inputs/world_points_coord.csv (id,x,y,z)
    """
    if not path.exists():
        raise FileNotFoundError(f"world_points_coord.csv not found at: {path}")
    m = {}
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = str(row["id"]).strip()
            x = float(row["x"]); y = float(row["y"])
            z = float(row.get("z", 0.0)) if row.get("z", "") != "" else 0.0
            m[pid] = (x, y, z)
    return m

def export_keypoints_inputs(world_csv_path: Path, out_world: Path, out_image: Path):
    """
    Creates:
      - inputs/keypoints_image.csv  (id,u,v) for V1,V3,V4,V5
      - inputs/keypoints_world.csv  (id,x,y,z) read from inputs/world_points_coord.csv for the same IDs
    """
    need = ["V1", "V3", "V4", "V5"]

    # image
    rows_img = []
    for pid in need:
        u, v = get_point(pid)  # can be manual or derived (V3/V4 are typically derived)
        rows_img.append((pid, u, v))

    # world
    wmap = read_world_points_map(world_csv_path)
    rows_w = []
    missing = [pid for pid in need if pid not in wmap]
    if missing:
        raise RuntimeError(f"Missing metric coordinates in world_points_coord.csv for: {missing}")
    for pid in need:
        x, y, z = wmap[pid]
        rows_w.append((pid, x, y, z))

    out_image.parent.mkdir(parents=True, exist_ok=True)
    with out_image.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "u", "v"])
        for pid, u, v in rows_img:
            w.writerow([pid, f"{u:.6f}", f"{v:.6f}"])
    print(f"[DONE] inputs/keypoints_image.csv: {out_image}")

    out_world.parent.mkdir(parents=True, exist_ok=True)
    with out_world.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "z"])
        for pid, x, y, z in rows_w:
            w.writerow([pid, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
    print(f"[DONE] inputs/keypoints_world.csv: {out_world}")


def final_save_all():
    """
    Should only be called at the end of the selection (after the final Save).
    Saves outputs and creates the additional inputs.
    """
    # outputs
    save_points_csv(OUT_POINTS_CSV)
    save_lines_csv(OUT_LINES_CSV)

    if not state["vps"]:
        raise RuntimeError("Vanishing points are not available yet. Press ENTER after finishing the lines to compute them.")
    save_vps_csv(OUT_VP_CSV)

    if state["conic"] is None:
        raise RuntimeError("Conic is not available yet. Press ENTER in STEP 4/4 (hoop) to fit the conic.")
    save_conic_csv(OUT_CONIC_CSV)

    # inputs (for subsequent scripts)
    export_inputs_image_points(INPUT_IMAGE_POINTS_CSV)
    export_inputs_vp_vertical(INPUT_VP_VERTICAL_TXT)
    export_keypoints_inputs(INPUT_WORLD_POINTS_CSV, INPUT_KEYPOINTS_WORLD, INPUT_KEYPOINTS_IMAGE)


# =============== STARTUP / REUSE ===============

def required_outputs_present():
    """
    There are now 4 required files to allow reuse:
      - points, lines, vanishing_points, conic
    """
    return (
        Path(OUT_POINTS_CSV).exists() and
        Path(OUT_LINES_CSV).exists() and
        Path(OUT_VP_CSV).exists() and
        Path(OUT_CONIC_CSV).exists()
    )

def load_existing_points_csv(path):
    pts = {}
    dpts = {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = str(row["id"]).strip()
            u = float(row["u"])
            v = float(row["v"])
            src = str(row.get("source", "")).strip().lower()
            if src == "manual_point":
                pts[pid] = (u, v)
            else:
                dpts[pid] = (u, v)
    return pts, dpts

def load_existing_lines_csv(path):
    lines = {}
    dlines = {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            lid = str(row["line_id"]).strip()
            typ = str(row["type"]).strip().lower()
            a = float(row["a"]); b = float(row["b"]); c = float(row["c"])
            h = (a, b, c)
            if typ == "manual":
                x1 = float(row["x1"]); y1 = float(row["y1"])
                x2 = float(row["x2"]); y2 = float(row["y2"])
                lines[lid] = {"p1": (x1, y1), "p2": (x2, y2), "h": h}
            else:
                dlines[lid] = h
    return lines, dlines

def load_existing_vps_csv(path):
    vps = {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            vid = str(row["vp_id"]).strip()
            vx = float(row["vx"]); vy = float(row["vy"]); vw = float(row["vw"])
            vps[vid] = (vx, vy, vw)
    return vps

def load_existing_conic_csv(path):
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
        if not rows:
            return None, None
        row = rows[0]
        a = float(row["a"]); b = float(row["b"]); c = float(row["c"])
        d = float(row["d"]); e = float(row["e"]); f0 = float(row["f"])
        conic = (a, b, c, d, e, f0)

        cu = row.get("center_u", "").strip()
        cv = row.get("center_v", "").strip()
        center = None
        if cu != "" and cv != "":
            center = (float(cu), float(cv))
        return conic, center

def load_everything_from_outputs():
    pts, dpts = load_existing_points_csv(OUT_POINTS_CSV)
    lines, dlines = load_existing_lines_csv(OUT_LINES_CSV)
    vps = load_existing_vps_csv(OUT_VP_CSV)
    conic, center = load_existing_conic_csv(OUT_CONIC_CSV)

    state["pts"] = pts
    state["derived_points"] = dpts
    state["lines"] = lines
    state["derived_lines"] = dlines
    state["vps"] = vps

    state["conic"] = conic
    state["conic_center"] = center

    if center is not None:
        state["derived_points"]["V3"] = center

    state["stage"] = "done"
    state["view_only"] = True
    state["startup_mode"] = "reuse"

def reset_for_manual_selection():
    state["stage"] = "points"
    state["view_only"] = False
    state["startup_mode"] = "manual"

    state["pts"] = {}
    state["lines"] = {}
    state["derived_lines"] = {}
    state["derived_points"] = {}
    state["vps"] = {}
    state["hoop_pts"] = []
    state["conic"] = None
    state["conic_center"] = None

    state["i_point"] = 0
    state["i_line"] = 0
    state["pending_p1"] = None


# =============== UI CALLBACKS ===============

def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # initial prompt: ignore clicks
    if state.get("startup_mode") == "prompt":
        return
    # view-only: ignore clicks
    if state.get("view_only", False):
        return

    # STEP 1: direct points
    if state["stage"] == "points":
        if state["i_point"] >= len(DIRECT_POINTS):
            return
        pid = DIRECT_POINTS[state["i_point"]]
        state["pts"][pid] = (x, y)
        state["i_point"] += 1
        if state["i_point"] >= len(DIRECT_POINTS):
            state["stage"] = "lines"
        redraw()
        return

    # STEP 2: manual lines (2 clicks per line)
    if state["stage"] == "lines":
        if state["i_line"] >= len(MANUAL_LINES):
            return
        lid = MANUAL_LINES[state["i_line"]]

        if state["pending_p1"] is None:
            state["pending_p1"] = (x, y)
        else:
            p1 = state["pending_p1"]
            p2 = (x, y)
            state["pending_p1"] = None
            state["lines"][lid] = {"p1": p1, "p2": p2, "h": segment_to_line_h(p1, p2)}
            state["i_line"] += 1
        redraw()
        return

    # STEP 4: hoop points
    if state["stage"] == "hoop":
        state["hoop_pts"].append((float(x), float(y)))
        redraw()
        return


def undo():
    # view-only or prompt: no undo
    if state.get("startup_mode") == "prompt" or state.get("view_only", False):
        return

    if state["stage"] == "hoop":
        if state["hoop_pts"]:
            state["hoop_pts"].pop()
            state["conic"] = None
            state["conic_center"] = None
            state["derived_points"].pop("V3", None)
            redraw()
        return

    if state["stage"] == "lines":
        if state["pending_p1"] is not None:
            state["pending_p1"] = None
            redraw()
            return
        if state["i_line"] > 0:
            state["i_line"] -= 1
            lid = MANUAL_LINES[state["i_line"]]
            state["lines"].pop(lid, None)
            redraw()
            return
        state["stage"] = "points"
        redraw()
        return

    if state["stage"] == "points":
        if state["i_point"] > 0:
            state["i_point"] -= 1
            pid = DIRECT_POINTS[state["i_point"]]
            state["pts"].pop(pid, None)
            redraw()
        return

    if state["stage"] == "done":
        state["stage"] = "hoop"
        state["conic"] = None
        state["conic_center"] = None
        state["hoop_pts"] = []
        state["derived_points"].pop("V3", None)
        redraw()


# =============== MAIN ===============

def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Could not read: {IMAGE_PATH}")

    state["img"] = img

    # STARTUP
    if required_outputs_present():
        state["startup_mode"] = "prompt"
        state["view_only"] = False
        state["stage"] = "points"
    else:
        state["startup_mode"] = "manual"
        state["view_only"] = False
        state["stage"] = "points"

    redraw()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        cv2.imshow(WINDOW_NAME, state["canvas"])
        k = cv2.waitKey(20) & 0xFF

        if k == 27:  # ESC
            break

        # initial prompt decision
        if state.get("startup_mode") == "prompt":
            if k in (ord('y'), ord('Y')):
                reset_for_manual_selection()
                redraw()
                continue
            if k in (ord('n'), ord('N')):
                load_everything_from_outputs()
                redraw()
                continue

        elif k in (ord('u'), ord('U')):
            undo()

        elif k in (ord('s'), ord('S')):
            if state.get("view_only", False):
                print("[INFO] View-only: will not generate outputs/inputs (they already exist).")
            else:
                try:
                    final_save_all()
                except Exception as e:
                    print(f"[ERROR] Save failed: {e}")

        elif k in (13, 10):  # ENTER
            if state.get("view_only", False):
                continue

            # ENTER after finishing lines: compute and move to hoop
            if state["stage"] == "lines" and state["i_line"] >= len(MANUAL_LINES):
                compute_all()
                state["stage"] = "hoop"
                redraw()

            # ENTER in hoop stage: fit conic and estimate center
            elif state["stage"] == "hoop":
                if len(state["hoop_pts"]) < HOOP_MIN_PTS:
                    pass
                else:
                    fit_hoop_conic_and_center()
                    state["stage"] = "done"
                    redraw()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
