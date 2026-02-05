"""
use_calibration_ground.py

Lê a calibração refinada (outputs/calibration_refined.json) e fornece:
- project_ground(X, Y): projeta (X,Y,0) -> (u,v) (com distorção k1)
- pixel_to_ground(u, v): (u,v) -> (X,Y,0) por interseção raio-plano Z=0
- opcional: desenhar grelha métrica no frame para validação visual

Requisitos: numpy, opencv-python
"""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import cv2

# ---------------- CONFIG ----------------
CALIB_JSON = Path("outputs/calibration_refined.json")
IMAGE_PATH = Path("data/new_dataset/images/shot_1_f0000.jpg")
OUT_OVERLAY = Path("outputs") / "overlay_grid.png"

# Grelha (metros)
GRID_X_MIN, GRID_X_MAX, GRID_X_STEP = -1.0, 10.0, 1.0
GRID_Y_MIN, GRID_Y_MAX, GRID_Y_STEP = -2.0,  8.0, 1.0
# --------------------------------------


def load_calibration(calib_json: Path):
    data = json.loads(calib_json.read_text(encoding="utf-8"))

    K = np.array(data["refined"]["K"], dtype=float)
    dist = np.array(data["refined"]["dist"], dtype=float).reshape(-1)

    R = np.array(data["refined"]["R_world_to_cam"], dtype=float)
    t = np.array(data["refined"]["t_world_to_cam"], dtype=float).reshape(3)

    W = int(data["image_size"]["width"])
    H = int(data["image_size"]["height"])

    # dist no formato do OpenCV para project/undistortPoints
    # Aqui usamos só [k1,k2,p1,p2,k3] (o teu script fixou os restantes a 0)
    if dist.size == 5:
        dist_cv = dist.reshape(5, 1)
    else:
        # fallback: pelo menos k1
        dist_cv = np.zeros((5, 1), dtype=float)
        dist_cv[0, 0] = float(dist[0]) if dist.size > 0 else 0.0

    rvec, _ = cv2.Rodrigues(R)

    return {
        "K": K,
        "dist": dist_cv,
        "R": R,
        "t": t,
        "rvec": rvec,
        "W": W,
        "H": H,
    }


def project_ground(X: float, Y: float, calib: dict) -> tuple[float, float]:
    """
    Projeta o ponto do chão (X,Y,0) para o pixel (u,v), usando K, distorção e (R,t).
    """
    obj = np.array([[X, Y, 0.0]], dtype=np.float64)
    uv, _ = cv2.projectPoints(obj, calib["rvec"], calib["t"].reshape(3, 1), calib["K"], calib["dist"])
    u, v = uv.reshape(2)
    return float(u), float(v)


def pixel_to_ground(u: float, v: float, calib: dict) -> tuple[float, float]:
    """
    Faz back-projection: pixel (u,v) -> ponto no plano Z=0 do mundo.
    Passos:
      1) UndistortPoints: (u,v) -> coordenadas normalizadas (x_n, y_n) no referencial da câmara
      2) Raio em câmara: d_c = [x_n, y_n, 1]^T
      3) Converter raio para mundo: d_w = R^T d_c
      4) Centro da câmara no mundo: C = -R^T t
      5) Interseção com plano Z=0: C_z + s d_wz = 0 => s = -C_z / d_wz
      6) P = C + s d_w => (X,Y)
    """
    K = calib["K"]
    dist = calib["dist"]
    R = calib["R"]
    t = calib["t"]

    # 1) UndistortPoints devolve coordenadas "normalizadas" (sem K): (x_n, y_n)
    pts = np.array([[[u, v]]], dtype=np.float64)
    und = cv2.undistortPoints(pts, K, dist, P=None)  # P=None -> normalizado
    x_n, y_n = und.reshape(2)

    # 2) direção no referencial da câmara
    d_c = np.array([x_n, y_n, 1.0], dtype=float)

    # 3) converter para mundo
    d_w = R.T @ d_c

    # 4) centro da câmara no mundo
    C = -R.T @ t

    # 5) interseção com Z=0
    if abs(d_w[2]) < 1e-12:
        raise RuntimeError("Raio quase paralelo ao plano Z=0 (d_wz ~ 0).")

    s = -C[2] / d_w[2]
    P = C + s * d_w
    return float(P[0]), float(P[1])


def camera_center_in_world(calib: dict) -> np.ndarray:
    R = calib["R"]
    t = calib["t"]
    C = -R.T @ t
    return C


def draw_grid_overlay(image_path: Path, calib: dict, out_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Não consegui ler a imagem: {image_path}")

    H_img, W_img = img.shape[:2]

    # Desenhar linhas X constantes (varia Y)
    xs = np.arange(GRID_X_MIN, GRID_X_MAX + 1e-9, GRID_X_STEP)
    ys = np.arange(GRID_Y_MIN, GRID_Y_MAX + 1e-9, GRID_Y_STEP)

    # helper para desenhar polilinha se os pontos estiverem dentro/razoáveis
    # def try_polyline(pts_uv):
    #     pts = np.array(pts_uv, dtype=np.float32).reshape(-1, 1, 2)
    #     # desenha sempre, mas podes filtrar por bounds se quiseres
    #     cv2.polylines(img, [pts], isClosed=False, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    def try_polyline(pts_uv):
        """
        Desenha a polilinha apenas com pontos válidos (finitos e com coordenadas razoáveis).
        Se houver quebras (pontos inválidos), desenha por segmentos.
        """
        # filtra finitos
        pts_uv = np.asarray(pts_uv, dtype=np.float64)

        finite = np.isfinite(pts_uv).all(axis=1)

        # também filtra valores absurdos (evita overflow no OpenCV)
        # permite um pouco fora da imagem para manter continuidade
        margin = 2000.0
        in_range = (
            (pts_uv[:, 0] > -margin) & (pts_uv[:, 0] < (W_img - 1 + margin)) &
            (pts_uv[:, 1] > -margin) & (pts_uv[:, 1] < (H_img - 1 + margin))
        )

        ok = finite & in_range

        # desenhar por segmentos contínuos de pontos ok
        start = None
        for i in range(len(pts_uv) + 1):
            if i < len(pts_uv) and ok[i]:
                if start is None:
                    start = i
            else:
                if start is not None:
                    seg = pts_uv[start:i]
                    if seg.shape[0] >= 2:
                        pts = seg.astype(np.int32).reshape(-1, 1, 2)
                        cv2.polylines(img, [pts], isClosed=False, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
                start = None

    # linhas verticais do grid (X fixo)
    for X in xs:
        pts_uv = []
        for Y in ys:
            u, v = project_ground(float(X), float(Y), calib)
            pts_uv.append([u, v])
        try_polyline(pts_uv)

    # linhas horizontais do grid (Y fixo)
    for Y in ys:
        pts_uv = []
        for X in xs:
            u, v = project_ground(float(X), float(Y), calib)
            pts_uv.append([u, v])
        try_polyline(pts_uv)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"[DONE] Overlay guardado em: {out_path}")


def main():
    calib = load_calibration(CALIB_JSON)

    C = camera_center_in_world(calib)
    print(f"[INFO] Centro da câmara no mundo C = {C} (espera-se C_z > 0)")

    # Exemplos rápidos
    u0, v0 = project_ground(0.0, 0.0, calib)
    print(f"[EX] (X,Y,0)=(0,0,0) -> (u,v)=({u0:.2f},{v0:.2f})")

    Xg, Yg = pixel_to_ground(u0, v0, calib)
    print(f"[EX] (u,v)=({u0:.2f},{v0:.2f}) -> (X,Y)=({Xg:.4f},{Yg:.4f})")

    # Overlay de grelha (validação visual)
    if IMAGE_PATH.exists():
        draw_grid_overlay(IMAGE_PATH, calib, OUT_OVERLAY)
    else:
        print(f"[WARN] IMAGE_PATH não existe: {IMAGE_PATH} (saltei overlay)")


if __name__ == "__main__":
    main()