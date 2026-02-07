import cv2
import numpy as np

VIDEO_PATH = "data/videos/Basket_free_shots_3.mp4"

# caixa inicial a partir do clique
INIT_BOX_W = 70
INIT_BOX_H = 70

# janela de reaquisição (em torno da previsão do tracker)
SEARCH_RADIUS = 140

# filtros do blob (ajusta depois)
MIN_R = 6
MAX_R = 45
MIN_CIRC = 0.75  # circularidade mínima

TRAIL_LEN = 140

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def make_box_from_click(x, y, w, h, W, H):
    x1 = clamp(int(x - w/2), 0, W-1)
    y1 = clamp(int(y - h/2), 0, H-1)
    x2 = clamp(int(x1 + w), 1, W)
    y2 = clamp(int(y1 + h), 1, H)
    return (x1, y1, x2 - x1, y2 - y1)

def new_csrt():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    return cv2.legacy.TrackerCSRT_create()

def crop_roi(frame, cx, cy, r):
    H, W = frame.shape[:2]
    x1 = clamp(int(cx - r), 0, W-1)
    y1 = clamp(int(cy - r), 0, H-1)
    x2 = clamp(int(cx + r), 1, W)
    y2 = clamp(int(cy + r), 1, H)
    return frame[y1:y2, x1:x2], (x1, y1)

def circularity(cnt):
    area = cv2.contourArea(cnt)
    per = cv2.arcLength(cnt, True)
    if per <= 1e-6:
        return 0.0
    return 4.0 * np.pi * area / (per * per)

def reacquire_dark_blob(frame, pred_cx, pred_cy):
    """
    Procura um blob escuro ~circular numa janela local.
    Retorna box (x,y,w,h) e centro (cx,cy) em coords globais, ou None.
    """
    roi, (ox, oy) = crop_roi(frame, pred_cx, pred_cy, SEARCH_RADIUS)
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # realçar “escuro” local: threshold adaptativo inverso
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )

    # limpar ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue

        (x, y, w, h) = cv2.boundingRect(cnt)
        r = 0.5 * (w + h) / 2.0  # raio aproximado
        if r < MIN_R or r > MAX_R:
            continue

        circ = circularity(cnt)
        if circ < MIN_CIRC:
            continue

        cx = ox + x + w/2
        cy = oy + y + h/2

        # score: preferir mais circular e mais perto da previsão
        dist = np.hypot(cx - pred_cx, cy - pred_cy)
        score = 2.0*circ - 0.01*dist + 0.0001*area

        if score > best_score:
            best_score = score
            best = (int(ox + x), int(oy + y), int(w), int(h), float(cx), float(cy), th, (ox, oy))

    return best  # inclui máscara para debug

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não consegui abrir o vídeo: {VIDEO_PATH}")

    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Não consegui ler o primeiro frame.")

    H, W = frame0.shape[:2]
    click = {"pt": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["pt"] = (x, y)

    cv2.namedWindow("Click BALL then press C")
    cv2.setMouseCallback("Click BALL then press C", on_mouse)

    while True:
        disp = frame0.copy()
        if click["pt"] is not None:
            x, y = click["pt"]
            box = make_box_from_click(x, y, INIT_BOX_W, INIT_BOX_H, W, H)
            bx, by, bw, bh = box
            cv2.rectangle(disp, (bx, by), (bx+bw, by+bh), (0,255,0), 2)
            cv2.circle(disp, (x, y), 5, (0,255,0), -1)
            cv2.putText(disp, "C=confirm  R=reset  ESC=quit", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        else:
            cv2.putText(disp, "Click the BALL, then press C", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Click BALL then press C", disp)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cap.release()
            cv2.destroyAllWindows()
            return
        if k in (ord('r'), ord('R')):
            click["pt"] = None
        if k in (ord('c'), ord('C')) and click["pt"] is not None:
            break

    cv2.destroyWindow("Click BALL then press C")

    x, y = click["pt"]
    init_box = make_box_from_click(x, y, INIT_BOX_W, INIT_BOX_H, W, H)

    tracker = new_csrt()
    tracker.init(frame0, init_box)

    trail = []
    bx, by, bw, bh = init_box
    pred_cx, pred_cy = bx + bw/2, by + bh/2
    trail.append((pred_cx, pred_cy))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        okT, box = tracker.update(frame)
        annotated = frame.copy()

        debug_mask = None

        if okT:
            bx, by, bw, bh = box
            bx, by, bw, bh = int(bx), int(by), int(bw), int(bh)
            pred_cx, pred_cy = bx + bw/2, by + bh/2

            # tentativa de reaquisição “bola escura” perto da previsão
            reacq = reacquire_dark_blob(frame, pred_cx, pred_cy)
            if reacq is not None:
                rx, ry, rw, rh, rcx, rcy, th, (ox, oy) = reacq

                # se encontrarmos um candidato razoável, reinicializar tracker
                tracker = new_csrt()
                tracker.init(frame, (rx, ry, rw, rh))
                bx, by, bw, bh = rx, ry, rw, rh
                pred_cx, pred_cy = rcx, rcy
                debug_mask = th

                cv2.putText(annotated, "Reacquired", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            else:
                cv2.putText(annotated, "Tracking", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            trail.append((pred_cx, pred_cy))
            if len(trail) > TRAIL_LEN:
                trail = trail[-TRAIL_LEN:]

            cv2.rectangle(annotated, (bx, by), (bx+bw, by+bh), (0,255,0), 2)
            cv2.circle(annotated, (int(pred_cx), int(pred_cy)), 5, (0,255,0), -1)

            # desenhar janela de pesquisa
            cv2.circle(annotated, (int(pred_cx), int(pred_cy)), SEARCH_RADIUS, (255,255,0), 1)

        else:
            cv2.putText(annotated, "LOST (press ESC)", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        for i in range(1, len(trail)):
            p0 = (int(trail[i-1][0]), int(trail[i-1][1]))
            p1 = (int(trail[i][0]), int(trail[i][1]))
            cv2.line(annotated, p0, p1, (255,255,0), 2)

        cv2.imshow("Ball tracking + reacquire (ESC)", annotated)
        if debug_mask is not None:
            cv2.imshow("Reacquire mask (debug)", debug_mask)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
