"""
Script de diagnóstico para verificar:
1. Calibração está correta
2. Keypoints world estão bem posicionados
3. Projeção do rim está correta
"""

import json
import csv
import numpy as np
import cv2
from pathlib import Path

# ================== CONFIG ==================
CALIB_JSON = r"..\outputs\calibration_refined.json"
KEYPOINTS_WORLD_CSV = r"..\inputs\keypoints_world.csv"
IMAGE_PATH = r"..\data\new_dataset\images\shot_1_f0000.jpg"  # Frame de referência
RIM_ID = "V3"

# Testa diferentes offsets
OFFSETS_TO_TEST = [
    [0.0, 0.0, 0.0],
    [0.0, -0.2, 0.02],
    [0.0, -0.3, 0.0],
    [0.0, -0.1, 0.0],
    [0.1, -0.2, 0.02],
    [-0.1, -0.2, 0.02],
]

OUT_DIR = r"..\outputs\diagnostics"
# ===========================================


def load_calibration(calib_json_path: str):
    with open(calib_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ref = data.get("refined", data)
    K = np.array(ref["K"], dtype=float)
    dist = np.array(ref.get("dist", [0, 0, 0, 0, 0]), dtype=float).reshape(-1, 1)
    R = np.array(ref["R_world_to_cam"], dtype=float)
    t = np.array(ref["t_world_to_cam"], dtype=float).reshape(3, 1)
    if "C_cam_in_world" in ref:
        C = np.array(ref["C_cam_in_world"], dtype=float).reshape(3, 1)
    else:
        C = -R.T @ t
    return {"K": K, "dist": dist, "R_wc": R, "t_wc": t, "C_w": C}


def load_keypoints_world(csv_path: str):
    out = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = row["id"].strip()
            out[pid] = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=float)
    return out


def project_world_points(Pw, K, dist, R_wc, t_wc):
    Pw = np.asarray(Pw, dtype=float).reshape(-1, 3)
    rvec, _ = cv2.Rodrigues(R_wc)
    uv, _ = cv2.projectPoints(Pw, rvec, t_wc, K, dist)
    return uv.reshape(-1, 2)


def draw_rim_circle(img, uv_center, radius_px, label, color):
    """Desenha círculo do rim com label"""
    u, v = int(round(uv_center[0])), int(round(uv_center[1]))
    
    # Círculo
    cv2.circle(img, (u, v), int(radius_px), color, 2, cv2.LINE_AA)
    
    # Centro
    cv2.circle(img, (u, v), 8, color, -1, cv2.LINE_AA)
    cv2.circle(img, (u, v), 12, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Cruz
    cv2.drawMarker(img, (u, v), color, cv2.MARKER_CROSS, 25, 3)
    
    # Label
    cv2.putText(img, label, (u + 20, v - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(img, f"({u}, {v})", (u + 20, v + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def estimate_rim_radius_pixels(P_rim, rim_radius_m, K, dist, R_wc, t_wc):
    """
    Estima o raio do rim em pixels projetando um ponto na borda.
    Assume que o rim está paralelo ao plano XY (horizontal).
    """
    # Ponto no centro
    uv_center = project_world_points([P_rim], K, dist, R_wc, t_wc)[0]
    
    # Ponto na borda (offset em X)
    P_edge = P_rim + np.array([rim_radius_m, 0, 0])
    uv_edge = project_world_points([P_edge], K, dist, R_wc, t_wc)[0]
    
    # Distância em pixels
    radius_px = np.linalg.norm(uv_edge - uv_center)
    return radius_px


def main():
    print("=" * 70)
    print("DIAGNÓSTICO DE CALIBRAÇÃO E POSIÇÃO DO RIM")
    print("=" * 70)
    print()
    
    # Load data
    calib = load_calibration(CALIB_JSON)
    kp_w = load_keypoints_world(KEYPOINTS_WORLD_CSV)
    
    print("CALIBRAÇÃO CARREGADA:")
    print(f"  K (matriz intrínseca):")
    print(f"    fx={calib['K'][0,0]:.2f}, fy={calib['K'][1,1]:.2f}")
    print(f"    cx={calib['K'][0,2]:.2f}, cy={calib['K'][1,2]:.2f}")
    print(f"  Distorção: {calib['dist'].ravel()}")
    print(f"  Posição câmara (mundo): {calib['C_w'].ravel()}")
    print()
    
    print(f"KEYPOINTS CARREGADOS: {len(kp_w)} pontos")
    for pid, pos in kp_w.items():
        print(f"  {pid}: X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")
    print()
    
    if RIM_ID not in kp_w:
        print(f"❌ ERRO: Keypoint '{RIM_ID}' não encontrado!")
        return
    
    P_rim_base = kp_w[RIM_ID]
    print(f"RIM KEYPOINT '{RIM_ID}' (sem offset):")
    print(f"  Posição: X={P_rim_base[0]:.3f}, Y={P_rim_base[1]:.3f}, Z={P_rim_base[2]:.3f}")
    print()
    
    # Load image
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"❌ ERRO: Não consegui ler imagem: {IMAGE_PATH}")
        return
    
    h, w = img.shape[:2]
    print(f"IMAGEM: {w}x{h} pixels")
    print()
    
    # Create output directory
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Projeta o rim base (sem offset)
    uv_rim_base = project_world_points([P_rim_base], calib["K"], calib["dist"], 
                                       calib["R_wc"], calib["t_wc"])[0]
    
    print("PROJEÇÃO DO RIM (SEM OFFSET):")
    print(f"  Pixel: u={uv_rim_base[0]:.1f}, v={uv_rim_base[1]:.1f}")
    
    # Verifica se está dentro da imagem
    if 0 <= uv_rim_base[0] < w and 0 <= uv_rim_base[1] < h:
        print(f"  ✓ Dentro da imagem")
    else:
        print(f"  ❌ FORA DA IMAGEM! Isto é um problema sério!")
    print()
    
    # Estima raio em pixels
    rim_radius_m = 0.225
    radius_px = estimate_rim_radius_pixels(P_rim_base, rim_radius_m, 
                                           calib["K"], calib["dist"],
                                           calib["R_wc"], calib["t_wc"])
    print(f"RAIO DO RIM ESTIMADO: {radius_px:.1f} pixels (para raio={rim_radius_m}m no mundo)")
    print()
    
    # Testa cada offset
    print("=" * 70)
    print("TESTANDO DIFERENTES OFFSETS")
    print("=" * 70)
    print()
    
    colors = [
        (255, 0, 0),    # Azul
        (0, 255, 0),    # Verde
        (0, 0, 255),    # Vermelho
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Amarelo
    ]
    
    img_test = img.copy()
    
    for i, offset in enumerate(OFFSETS_TO_TEST):
        offset_arr = np.array(offset, dtype=float)
        P_rim_offset = P_rim_base + offset_arr
        
        uv_rim = project_world_points([P_rim_offset], calib["K"], calib["dist"],
                                      calib["R_wc"], calib["t_wc"])[0]
        
        radius_px_offset = estimate_rim_radius_pixels(P_rim_offset, rim_radius_m,
                                                      calib["K"], calib["dist"],
                                                      calib["R_wc"], calib["t_wc"])
        
        label = f"Offset{i}: [{offset[0]:.2f}, {offset[1]:.2f}, {offset[2]:.2f}]"
        color = colors[i % len(colors)]
        
        print(f"{label}")
        print(f"  Posição 3D: X={P_rim_offset[0]:.3f}, Y={P_rim_offset[1]:.3f}, Z={P_rim_offset[2]:.3f}")
        print(f"  Projeção 2D: u={uv_rim[0]:.1f}, v={uv_rim[1]:.1f}")
        print(f"  Raio (pixels): {radius_px_offset:.1f}")
        
        if 0 <= uv_rim[0] < w and 0 <= uv_rim[1] < h:
            print(f"  ✓ Dentro da imagem")
        else:
            print(f"  ❌ Fora da imagem")
        print()
        
        # Desenha no overlay
        draw_rim_circle(img_test, uv_rim, radius_px_offset, label, color)
    
    # Salva imagem de teste
    out_path = Path(OUT_DIR) / "rim_offset_test.png"
    cv2.imwrite(str(out_path), img_test)
    print(f"✓ Imagem de teste guardada: {out_path}")
    print()
    
    # Projeta TODOS os keypoints para verificar calibração
    print("=" * 70)
    print("PROJEÇÃO DE TODOS OS KEYPOINTS (verificar calibração)")
    print("=" * 70)
    print()
    
    img_all_kp = img.copy()
    
    for pid, P_w in kp_w.items():
        uv = project_world_points([P_w], calib["K"], calib["dist"],
                                  calib["R_wc"], calib["t_wc"])[0]
        
        u, v = int(round(uv[0])), int(round(uv[1]))
        
        print(f"{pid}: 3D=({P_w[0]:.2f}, {P_w[1]:.2f}, {P_w[2]:.2f}) → 2D=({u}, {v})", end="")
        
        if 0 <= u < w and 0 <= v < h:
            print(" ✓")
            cv2.circle(img_all_kp, (u, v), 8, (0, 255, 0), -1)
            cv2.circle(img_all_kp, (u, v), 12, (255, 255, 255), 2)
            cv2.putText(img_all_kp, pid, (u + 15, v - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            print(" ❌ FORA!")
    
    print()
    out_path_kp = Path(OUT_DIR) / "all_keypoints_projected.png"
    cv2.imwrite(str(out_path_kp), img_all_kp)
    print(f"✓ Projeção de todos os keypoints: {out_path_kp}")
    print()
    
    # Instruções finais
    print("=" * 70)
    print("PRÓXIMOS PASSOS")
    print("=" * 70)
    print()
    print("1. Abre 'rim_offset_test.png' e verifica qual offset coloca o centro")
    print("   do rim (círculo colorido) EXATAMENTE no centro do aro na imagem.")
    print()
    print("2. Abre 'all_keypoints_projected.png' e verifica se TODOS os pontos")
    print("   verdes estão nos locais corretos da imagem.")
    print("   - Se não estiverem: problema na CALIBRAÇÃO")
    print("   - Se estiverem mas o rim está errado: problema no KEYPOINT V3")
    print()
    print("3. Copia o offset correto para o código:")
    print("   RIM_OFFSET_W = np.array([dx, dy, dz], dtype=float)")
    print()
    print("4. Se nenhum offset funcionar, o problema está no keypoint V3 no CSV.")
    print("   Precisas medir manualmente a posição correta do centro do aro.")
    print()


if __name__ == "__main__":
    main()