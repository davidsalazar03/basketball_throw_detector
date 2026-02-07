import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "data/videos/Basket_free_shots_compilation.mp4"
MODEL_PATH = "yolov8n.pt"
BALL_CLASS_ID = 32  # "sports ball"

CONF_THRES = 0.20          # baixa um pouco porque a bola pode ser pequena
MAX_DETECTIONS = 50
MAX_JUMP_PX = 120          # limite de salto entre frames (ajusta depois)
TRAIL_LEN = 60             # nº de pontos da trajetória desenhada

def pick_ball_detection(result, last_xy=None):
    """
    Escolhe a deteção de bola 'mais plausível' no frame.
    Estratégia:
      - filtra só class=BALL_CLASS_ID
      - se não houver last_xy: escolhe maior confiança
      - se houver last_xy: escolhe a mais próxima, penalizando saltos grandes
    Retorna (cx, cy, conf) ou None
    """
    if result.boxes is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()

    idx = np.where(cls == BALL_CLASS_ID)[0]
    if idx.size == 0:
        return None

    # centros
    cxy = np.column_stack(((xyxy[idx, 0] + xyxy[idx, 2]) * 0.5,
                           (xyxy[idx, 1] + xyxy[idx, 3]) * 0.5))
    cconf = conf[idx]

    if last_xy is None:
        j = int(np.argmax(cconf))
        return float(cxy[j, 0]), float(cxy[j, 1]), float(cconf[j])

    last = np.array(last_xy, dtype=float).reshape(1, 2)
    d = np.linalg.norm(cxy - last, axis=1)

    # se tudo estiver muito longe, rejeita (provável falso positivo)
    j = int(np.argmin(d))
    if d[j] > MAX_JUMP_PX:
        return None

    return float(cxy[j, 0]), float(cxy[j, 1]), float(cconf[j])

def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não consegui abrir o vídeo: {VIDEO_PATH}")

    trail = []      # lista de (x,y)
    last_xy = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model.predict(frame, conf=CONF_THRES, max_det=MAX_DETECTIONS, verbose=False)[0]
        pick = pick_ball_detection(res, last_xy=last_xy)

        annotated = res.plot()

        if pick is not None:
            x, y, c = pick
            last_xy = (x, y)
            trail.append((x, y))
            if len(trail) > TRAIL_LEN:
                trail = trail[-TRAIL_LEN:]

            # marca centro da bola
            cv2.circle(annotated, (int(x), int(y)), 6, (0, 255, 0), -1)
            cv2.putText(annotated, f"ball conf={c:.2f}", (int(x)+10, int(y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # desenhar trajetória
        for i in range(1, len(trail)):
            p0 = (int(trail[i-1][0]), int(trail[i-1][1]))
            p1 = (int(trail[i][0]), int(trail[i][1]))
            cv2.line(annotated, p0, p1, (255, 255, 0), 2)

        cv2.imshow("Ball 2D track (ESC to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
