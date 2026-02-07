import cv2
import numpy as np

VIDEO_PATH = "data/videos/Basket_free_shots_1.mp4"

# parâmetros (vamos ajustar depois)
ROI = None                 # ex.: (x1, y1, x2, y2) se quiseres fixar
MIN_AREA = 30              # área mínima do blob (px^2)
MAX_AREA = 3000            # área máxima do blob
MAX_JUMP_PX = 150          # rejeitar saltos grandes
TRAIL_LEN = 80

def crop(frame, roi):
    if roi is None:
        return frame, (0, 0)
    x1, y1, x2, y2 = roi
    return frame[y1:y2, x1:x2], (x1, y1)

def pick_blob(contours, last_xy=None, offset=(0,0)):
    """
    Escolhe o blob mais plausível.
    - filtra por área
    - escolhe o mais próximo do último ponto, senão o maior
    """
    cands = []
    ox, oy = offset
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx = ox + x + w/2
        cy = oy + y + h/2
        cands.append((cx, cy, area, (ox+x, oy+y, w, h)))

    if not cands:
        return None

    if last_xy is None:
        # maior blob
        return max(cands, key=lambda t: t[2])

    lx, ly = last_xy
    # blob mais próximo
    cands.sort(key=lambda t: (t[0]-lx)**2 + (t[1]-ly)**2)
    best = cands[0]
    d = np.hypot(best[0]-lx, best[1]-ly)
    if d > MAX_JUMP_PX:
        return None
    return best

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não consegui abrir o vídeo: {VIDEO_PATH}")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Não consegui ler o primeiro frame.")

    # preparar background subtractor (robusto a iluminação)
    backsub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=False)

    trail = []
    last_xy = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        view, offset = crop(frame, ROI)

        gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        fg = backsub.apply(gray)

        # limpar ruído
        fg = cv2.medianBlur(fg, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, kernel, iterations=1)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pick = pick_blob(contours, last_xy=last_xy, offset=offset)

        annotated = frame.copy()

        # desenhar ROI se existir
        if ROI is not None:
            x1,y1,x2,y2 = ROI
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (255,255,0), 2)

        if pick is not None:
            cx, cy, area, (x, y, w, h) = pick
            last_xy = (cx, cy)
            trail.append((cx, cy))
            if len(trail) > TRAIL_LEN:
                trail = trail[-TRAIL_LEN:]

            cv2.rectangle(annotated, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)
            cv2.circle(annotated, (int(cx), int(cy)), 5, (0,255,0), -1)
            cv2.putText(annotated, f"area={int(area)}", (int(cx)+10, int(cy)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # trajetória
        for i in range(1, len(trail)):
            p0 = (int(trail[i-1][0]), int(trail[i-1][1]))
            p1 = (int(trail[i][0]), int(trail[i][1]))
            cv2.line(annotated, p0, p1, (255,255,0), 2)

        # debug: mostrar máscara de movimento
        fg_vis, off = crop(annotated, ROI)
        # janela principal
        cv2.imshow("Ball motion track (ESC)", annotated)
        cv2.imshow("Foreground mask", fg)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
