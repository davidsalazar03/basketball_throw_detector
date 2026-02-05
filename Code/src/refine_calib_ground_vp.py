"""
refine_calib_ground_vp.py

Refinamento (bundle adjustment simples) a partir de:
- H_ground_clean (mundo XY no chão -> imagem)
- vp_vertical (ponto de fuga vertical)
- inliers (IDs do ajuste da homografia)
- correspondências (X,Y,0) <-> (u,v)

Objetivo: reduzir o RMSE do modelo pinhole ajustando intrínsecos e extrínsecos.

Hipóteses (mantém "método da disciplina", mas com refinamento por reprojeção):
- skew = 0 (fixo)
- fx = fy (fixo via aspect ratio)
- principal point (cx,cy) livre (ajustável)
- distorção: por defeito estima só k1 (fixa k2..k6 e tangencial)
  (podes desligar com ESTIMATE_K1 = False)

Requisitos: numpy, pandas, opencv-python
"""

from __future__ import annotations

from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
import cv2


# ---------------- CONFIG ----------------
BASE_DIR = Path("inputs")
OUT_DIR = Path("outputs")

WORLD_CSV = BASE_DIR / "world_points_coord.csv"  # id,x,y,z
IMAGE_CSV = BASE_DIR / "image_points.csv"        # id,u,v
IMAGE_SIZE_TXT = BASE_DIR / "image_size.txt"     # W H
VP_VERTICAL_TXT = BASE_DIR / "vp_vertical.txt"   # u,v,w ou u v w

# Usa os outputs do script anterior:
H_CLEAN_TXT = OUT_DIR / "H_ground_clean.txt"
INLIERS_IDS_TXT = OUT_DIR / "inliers_ids.txt"

# refinamento
ESTIMATE_K1 = True      # True: estima k1; False: distorção = 0
MAX_ITERS = 300
EPS = 1e-10

OUT_JSON = OUT_DIR / "calibration_refined.json"
OUT_NPZ = OUT_DIR / "calibration_refined.npz"
# --------------------------------------


def read_image_size(path: Path) -> tuple[int, int]:
    txt = path.read_text(encoding="utf-8").strip()
    parts = txt.split()
    if len(parts) != 2:
        raise ValueError(f"Formato inválido em {path}. Esperado: 'W H'.")
    return int(parts[0]), int(parts[1])


def read_vp_vertical(path: Path) -> np.ndarray:
    txt = path.read_text(encoding="utf-8").strip()
    parts = re.split(r"[,\s]+", txt)
    parts = [p for p in parts if p != ""]
    if len(parts) != 3:
        raise ValueError(f"Formato inválido em {path}. Esperado: 3 números (u,v,w).")
    v = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)
    if abs(v[2]) > 1e-12:
        v = v / v[2]
    return v


def load_matrix_txt(path: Path) -> np.ndarray:
    """
    Lê matriz 3x3 de um TXT, aceitando:
    - linhas com espaços
    - linhas com vírgulas
    - linhas com brackets tipo [a b c]
    """
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    vals = []
    for ln in lines:
        ln = ln.replace("[", " ").replace("]", " ")
        parts = re.split(r"[,\s]+", ln)
        parts = [p for p in parts if p != ""]
        vals.append([float(x) for x in parts])
    M = np.array(vals, dtype=float)
    if M.shape != (3, 3):
        raise ValueError(f"Esperava 3x3 em {path}, obtive {M.shape}.")
    if abs(M[2, 2]) > 1e-12:
        M = M / M[2, 2]
    return M


def load_inlier_ids(path: Path) -> set[str]:
    ids = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return set(ids)


def load_points(world_csv: Path, image_csv: Path) -> pd.DataFrame:
    wdf = pd.read_csv(world_csv)
    idf = pd.read_csv(image_csv)

    for col in ["id", "x", "y"]:
        if col not in wdf.columns:
            raise ValueError(f"{world_csv} tem de conter '{col}'.")
    if "z" not in wdf.columns:
        wdf["z"] = 0.0

    for col in ["id", "u", "v"]:
        if col not in idf.columns:
            raise ValueError(f"{image_csv} tem de conter '{col}'.")
    wdf["id"] = wdf["id"].astype(str).str.strip()
    idf["id"] = idf["id"].astype(str).str.strip()

    df = wdf.merge(idf, on="id", how="inner")
    return df


def estimate_f_from_orthogonal_vps(vps: list[np.ndarray], W: int, H: int) -> tuple[float, float, float, list[float]]:
    """
    Assume: skew=0, fx=fy=f, principal point no centro (para seed).
    Para VPs ortogonais: (vi-c)·(vj-c) + f^2 = 0
    devolve f, cx, cy, e lista de f^2 por pares.
    """
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    c = np.array([cx, cy], dtype=float)
    pairs = [(0, 1), (0, 2), (1, 2)]
    f2_list = []
    for i, j in pairs:
        vi = vps[i][:2] - c
        vj = vps[j][:2] - c
        f2_list.append(-float(np.dot(vi, vj)))

    f2_pos = [x for x in f2_list if x > 0]
    if len(f2_pos) == 0:
        raise RuntimeError("Não obtive f^2 positivo a partir dos VPs com cx,cy no centro.")
    f = float(np.sqrt(np.median(np.array(f2_pos))))
    return f, cx, cy, f2_list


def decompose_H_to_pose(K: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    H = K [r1 r2 t] (até escala) -> R,t mundo->câmara
    """
    Kinv = np.linalg.inv(K)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    lam = 1.0 / (np.linalg.norm(Kinv @ h1) + 1e-12)
    r1 = lam * (Kinv @ h1)
    r2 = lam * (Kinv @ h2)
    r3 = np.cross(r1, r2)
    t = lam * (Kinv @ h3)
    R = np.column_stack([r1, r2, r3])

    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return R, t


def rmse_px(uv: np.ndarray, uv_hat: np.ndarray) -> float:
    d = uv - uv_hat
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def project_points_cv(obj_pts: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    uv_hat, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    return uv_hat.reshape(-1, 2)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    W, Himg = read_image_size(IMAGE_SIZE_TXT)
    v_z = read_vp_vertical(VP_VERTICAL_TXT)
    H_clean = load_matrix_txt(H_CLEAN_TXT)
    inliers = load_inlier_ids(INLIERS_IDS_TXT)

    df = load_points(WORLD_CSV, IMAGE_CSV)

    # Filtra chão (Z=0) e inliers
    df["z"] = df["z"].astype(float)
    df = df[np.abs(df["z"]) <= 1e-6].copy()
    df = df[df["id"].isin(inliers)].copy()
    if len(df) < 4:
        raise RuntimeError(f"Poucos inliers no chão para refinamento: N={len(df)}.")

    XY = df[["x", "y"]].to_numpy(dtype=float)
    uv = df[["u", "v"]].to_numpy(dtype=float)

    # VPs do plano a partir de H
    h1 = H_clean[:, 0].copy()
    h2 = H_clean[:, 1].copy()
    v_x = h1 / (h1[2] + 1e-12)
    v_y = h2 / (h2[2] + 1e-12)

    # Seed K0 (f, centro) e pose inicial
    f0, cx0, cy0, f2_list = estimate_f_from_orthogonal_vps([v_x, v_y, v_z], W, Himg)
    K0 = np.array([[f0, 0.0, cx0],
                   [0.0, f0, cy0],
                   [0.0, 0.0, 1.0]], dtype=float)

    R0, t0 = decompose_H_to_pose(K0, H_clean)
    rvec0, _ = cv2.Rodrigues(R0)
    tvec0 = t0.reshape(3, 1)

    # Distorção inicial
    dist0 = np.zeros((5, 1), dtype=np.float64)  # [k1,k2,p1,p2,k3]
    if not ESTIMATE_K1:
        dist0[:] = 0.0

    # RMSE inicial
    obj_pts = np.c_[XY, np.zeros((XY.shape[0], 1), dtype=float)].astype(np.float64)
    img_pts = uv.astype(np.float64)

    uv_hat0 = project_points_cv(obj_pts, rvec0, tvec0, K0, dist0)
    rmse0 = rmse_px(img_pts, uv_hat0)

    print(f"[SEED] f0={f0:.3f}px, (cx0,cy0)=({cx0:.1f},{cy0:.1f}), f^2 pares={f2_list}")
    print(f"[SEED] RMSE reprojeção (inliers) antes de refinar: {rmse0:.3f} px")

    # Refinamento com calibrateCamera (1 imagem)
    objpoints = [obj_pts.astype(np.float32)]
    imgpoints = [img_pts.astype(np.float32)]
    image_size = (W, Himg)


    K_init = K0.astype(np.float64)
    K_init[0, 1] = 0.0  # skew=0
    K_init[0,2] = (W - 1)/2
    K_init[1,2] = (Himg - 1)/2
    
    dist_init = dist0.astype(np.float64)

    flags = 0

    # helper para adicionar flags só se existirem
    def add_flag(name: str):
        nonlocal flags
        if hasattr(cv2, name):
            flags |= getattr(cv2, name)

    add_flag("CALIB_USE_INTRINSIC_GUESS")
    add_flag("CALIB_FIX_ASPECT_RATIO")     # força fx/fy a manter-se (igual se K_init tiver fx=fy)
    add_flag("CALIB_ZERO_TANGENT_DIST")    # p1=p2=0
    add_flag("CALIB_FIX_PRINCIPAL_POINT")   # fixa cx,cy (se quiseres deixar livre, não adicionas esta flag e garantimos cx0,cy0 no seed)

    # skew: algumas builds não têm CALIB_FIX_SKEW
    # Se existir, usa; se não existir, garantimos skew=0 no guess e não o libertamos.
    add_flag("CALIB_FIX_SKEW")

    if ESTIMATE_K1:
        add_flag("CALIB_FIX_K2")
        add_flag("CALIB_FIX_K3")
        add_flag("CALIB_FIX_K4")
        add_flag("CALIB_FIX_K5")
        add_flag("CALIB_FIX_K6")
    else:
        add_flag("CALIB_FIX_K1")
        add_flag("CALIB_FIX_K2")
        add_flag("CALIB_FIX_K3")
        add_flag("CALIB_FIX_K4")
        add_flag("CALIB_FIX_K5")
        add_flag("CALIB_FIX_K6")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, MAX_ITERS, EPS)

    rms, K_ref, dist_ref, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,     # objectPoints
    imgpoints,     # imagePoints
    image_size,    # imageSize (W,H)
    K_init,        # cameraMatrix
    dist_init,     # distCoeffs
    None,          # rvecs (output)
    None,          # tvecs (output)
    flags,         # flags
    criteria       # criteria
    )

    rvec_ref = rvecs[0]
    tvec_ref = tvecs[0]
    R_ref, _ = cv2.Rodrigues(rvec_ref)
    t_ref = tvec_ref.reshape(3)

    # RMSE final (reprojeção)
    uv_hat_ref = project_points_cv(obj_pts, rvec_ref, tvec_ref, K_ref, dist_ref)
    rmse1 = rmse_px(img_pts, uv_hat_ref)

    C_ref = -R_ref.T @ t_ref

    print(f"[REFINE] OpenCV rms reportado: {float(rms):.3f}")
    print(f"[REFINE] RMSE reprojeção (inliers) depois de refinar: {rmse1:.3f} px")
    print(f"[REFINE] K_ref=\n{K_ref}")
    print(f"[REFINE] dist_ref=[k1,k2,p1,p2,k3]= {dist_ref.reshape(-1).tolist()}")

    payload = {
        "image_size": {"width": int(W), "height": int(Himg)},
        "mode": {"estimate_k1": bool(ESTIMATE_K1), "fx_equals_fy": True, "skew_fixed": True, "tangential_fixed": True},
        "inputs": {
            "H_clean_path": str(H_CLEAN_TXT),
            "vp_vertical_path": str(VP_VERTICAL_TXT),
            "inliers_ids_path": str(INLIERS_IDS_TXT),
        },
        "seed": {
            "K0": K0.tolist(),
            "f2_pairs": [float(x) for x in f2_list],
            "rmse_px": rmse0,
        },
        "refined": {
            "K": K_ref.tolist(),
            "dist": dist_ref.reshape(-1).tolist(),
            "R_world_to_cam": R_ref.tolist(),
            "t_world_to_cam": t_ref.tolist(),
            "C_cam_in_world": C_ref.tolist(),
            "rmse_px": rmse1,
            "opencv_rms": float(rms),
        }
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    np.savez(
        OUT_NPZ,
        K0=K0, R0=R0, t0=t0,
        K=K_ref, dist=dist_ref.reshape(-1),
        R=R_ref, t=t_ref, C=C_ref,
        rmse0=rmse0, rmse1=rmse1,
        v_x=v_x, v_y=v_y, v_z=v_z,
        H_clean=H_clean,
        ids=df["id"].to_numpy(dtype=object),
        XY=XY, uv=uv
    )

    print(f"[DONE] Guardado: {OUT_JSON}")
    print(f"[DONE] Guardado: {OUT_NPZ}")


if __name__ == "__main__":
    main()