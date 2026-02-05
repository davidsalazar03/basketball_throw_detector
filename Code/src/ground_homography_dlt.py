from pathlib import Path
import csv
import numpy as np
import cv2

# ---------------- CONFIG ----------------
WORLD_CSV = r"outputs/world_points_coord.csv"     # id,x,y,z (metros)
IMAGE_CSV = r"outputs/court_points_image.csv"  # id,u,v (pixels) (o teu output)

IMAGE_PATH = r"data/new_dataset/images/shot_1_f0000.jpg"
OUT_RECTIFIED = r"outputs/rectified_floor.png"

PX_PER_M = 120  # escala da retificação (p.ex. 120 px por metro) -> ajusta
MARGIN_M = 0.5  # margem em metros à volta do bounding box dos pontos
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
            z = float(row.get("z", 0.0))  # se não existir, assume 0
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
    devolve T (3x3) e pts_norm (N,2) tais que x_norm ~ T x
    """
    pts = np.asarray(pts, dtype=float)
    mean = pts.mean(axis=0)
    d = np.sqrt(((pts - mean) ** 2).sum(axis=1)).mean()
    if d < 1e-12:
        raise ValueError("Pontos degenerados na normalização.")

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
    Estima H tal que [u v 1]^T ~ H [X Y 1]^T
    via DLT com normalização (Hartley).
    world_xy: (N,2), image_uv: (N,2), N>=4
    """
    world_xy = np.asarray(world_xy, dtype=float)
    image_uv = np.asarray(image_uv, dtype=float)
    if world_xy.shape[0] < 4:
        raise ValueError("Precisas de pelo menos 4 correspondências no chão para homografia.")

    Tw, w_n = normalize_points_2d(world_xy)
    Ti, i_n = normalize_points_2d(image_uv)

    N = world_xy.shape[0]
    A = []
    for k in range(N):
        X, Y = w_n[k]
        u, v = i_n[k]
        # linhas do DLT
        A.append([0, 0, 0, -X, -Y, -1, v*X, v*Y, v])
        A.append([X, Y, 1,  0,  0,  0, -u*X, -u*Y, -u])
    A = np.asarray(A, dtype=float)

    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)

    # desnormalizar: i ~ Ti^{-1} Hn Tw
    H = np.linalg.inv(Ti) @ Hn @ Tw

    # normaliza para H[2,2]=1 (se possível)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def apply_homography(H, pts):
    """
    H: 3x3, pts: (N,2)
    devolve (N,2)
    """
    pts = np.asarray(pts, dtype=float)
    pts_h = np.c_[pts, np.ones((pts.shape[0], 1))]
    q = (H @ pts_h.T).T
    q = q[:, :2] / q[:, 2:3]
    return q


def main():
    world = read_world_points(WORLD_CSV)
    image = read_image_points(IMAGE_CSV)

    plane_type = "ground"  # MUDA AQUI: "ground" ou "vertical"

    # escolher apenas IDs comuns e filtrar por plano
    common = []
    world_coords = []  # vai conter (coord1, coord2) dependendo do plano
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
                # Para plano vertical em X=X0, usamos (Y, Z)
                world_coords.append((y, z))
                image_uv.append((u, v))

        print(f"[INFO] Plano VERTICAL (X={X0}m): {len(common)} correspondências")
        
    elif plane_type == "ground":
        EPS = 1e-3  

        for pid, (x, y, z) in world.items():
            if abs(z) > EPS:
                continue
            if pid in image:
                u, v = image[pid]
                common.append(pid)
                # Para plano do chão (Z=0), usamos (X, Y)
                world_coords.append((x, y))
                image_uv.append((u, v))

        print(f"[INFO] Plano CHÃO (Z=0): {len(common)} correspondências")

    else:
        raise ValueError(f"plane_type desconhecido: {plane_type}")

    world_pts = np.asarray(world_coords, dtype=float)
    image_uv = np.asarray(image_uv, dtype=float)

    if len(common) < 4:
        raise RuntimeError(f"Não há correspondências suficientes (>=4). Encontradas: {len(common)}")

    # Mostrar pontos usados
    print(f"[DEBUG] IDs usados: {common}")
    print(f"[DEBUG] Primeiras 3 correspondências mundo->imagem:")
    for i in range(min(3, len(common))):
        print(f"  {common[i]}: mundo{world_pts[i]} -> imagem{image_uv[i]}")

    # -------- estimar H --------
    if USE_RANSAC:
        # cv2.findHomography estima H (mundo->imagem) com RANSAC
        H_cv, mask = cv2.findHomography(world_pts, image_uv, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H_cv is None:
            raise RuntimeError("cv2.findHomography falhou.")
        H = H_cv
        inliers = int(mask.sum()) if mask is not None else len(common)
        print(f"[INFO] RANSAC inliers: {inliers}/{len(common)}")
    else:
        H = dlt_homography(world_pts, image_uv)

    print(f"[DEBUG] Homografia H (mundo->imagem):")
    print(H)

    Hinv = np.linalg.inv(H)

    # -------- retificar o plano para imagem --------
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Não consegui ler: {IMAGE_PATH}")

    # bounding box no mundo para definir o canvas métrico
    coord_min = world_pts.min(axis=0) - MARGIN_M
    coord_max = world_pts.max(axis=0) + MARGIN_M

    print(f"[DEBUG] Bounding box mundo: min={coord_min}, max={coord_max}")

    out_w = int(np.ceil((coord_max[0] - coord_min[0]) * PX_PER_M))
    out_h = int(np.ceil((coord_max[1] - coord_min[1]) * PX_PER_M))

    print(f"[INFO] Canvas de saída: {out_w} x {out_h} pixels")

    # Mapeamento "mundo->canvas_px"
    # Para ambos os planos, queremos:
    #   - coordenada horizontal (coord1) cresce para a direita: x_px = (coord1 - min1) * PX_PER_M
    #   - coordenada vertical (coord2) cresce para CIMA no mundo, mas pixels crescem para BAIXO
    #     então: y_px = (max2 - coord2) * PX_PER_M
    
    T = np.array([
        [PX_PER_M,  0,         -coord_min[0] * PX_PER_M],
        [0,         -PX_PER_M, coord_max[1] * PX_PER_M],
        [0,          0,         1]
    ], dtype=float)

    print(f"[DEBUG] Matriz de transformação mundo->canvas T:")
    print(T)

    # Queremos warpPerspective usando matriz que leve da imagem -> canvas.
    # imagem -> mundo: Hinv
    # mundo -> canvas: T
    # logo: imagem -> canvas = T @ Hinv
    W = T @ Hinv

    print(f"[DEBUG] Matriz final warp W (imagem->canvas):")
    print(W)

    rect = cv2.warpPerspective(img, W, (out_w, out_h))
    Path(OUT_RECTIFIED).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(OUT_RECTIFIED, rect)
    print(f"[DONE] Retificação guardada em: {OUT_RECTIFIED}")

    # -------- sanity check (erro médio reprojeção) --------
    proj_uv = apply_homography(H, world_pts)      # mundo->imagem
    err = np.linalg.norm(proj_uv - image_uv, axis=1)
    print(f"[CHECK] Erro reprojeção: médio={err.mean():.2f}px | max={err.max():.2f}px")
    
    # Mostrar erros individuais para debug
    print(f"[DEBUG] Erros individuais de reprojeção:")
    for i, pid in enumerate(common):
        print(f"  {pid}: {err[i]:.2f}px")

    H_GROUND_PATH = r"inputs/H_ground.txt"

    # Escreve (cria ou substitui o ficheiro inteiro)
    with open(H_GROUND_PATH, "a", encoding="utf-8") as f:
        f.write(H[0].__str__() + "\n")
        f.write(H[1].__str__() + "\n")
        f.write(H[2].__str__() + "\n")
    print(f"[DONE] Matriz de transformação mundo->canvas guardada em: {H_GROUND_PATH}")



if __name__ == "__main__":
    main()