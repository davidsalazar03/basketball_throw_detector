from pathlib import Path
import cv2
import numpy as np
import csv
from ultralytics import YOLO

#VIDEO_PATH = r"data/videos_raw/Basket_free_shots_4.mp4"   # <-- muda para o teu
#VIDEO_PATH = r"data/videos_raw/Basket_free_shots_compilation.mp4"   # <-- muda para o teu
VIDEO_PATH = r"data/new_videos_raw/shot_1.mp4"

MODEL_PATH = r"runs/detect/train3/weights/best.pt"

OUT_CSV = r"outputs/ball_traj.csv"
CONF = 0.20
IMG_SZ = 640
TRAIL_LEN = 80

def main():
    Path("outputs").mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não consegui abrir: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_id = 0
    trail = []

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "t_sec", "cx", "cy", "x1", "y1", "x2", "y2", "conf"])

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t_sec = frame_id / fps

            # inferência (no mesmo frame!)
            res = model.predict(frame, device=0, imgsz=IMG_SZ, conf=CONF, verbose=False)[0]

            annotated = frame.copy()

            if res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes
                confs = boxes.conf.detach().cpu().numpy()
                xyxy = boxes.xyxy.detach().cpu().numpy()

                j = int(np.argmax(confs))
                x1, y1, x2, y2 = xyxy[j]
                c = float(confs[j])

                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)

                trail.append((cx, cy))
                if len(trail) > TRAIL_LEN:
                    trail = trail[-TRAIL_LEN:]

                # desenhar bbox + centro
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(annotated, (int(cx), int(cy)), 4, (0, 255, 0), -1)
                cv2.putText(annotated, f"conf={c:.2f}", (int(x1), max(0, int(y1)-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # guardar CSV
                w.writerow([frame_id, f"{t_sec:.6f}", f"{cx:.2f}", f"{cy:.2f}",
                            f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}", f"{c:.4f}"])
            else:
                # sem deteção nesse frame
                w.writerow([frame_id, f"{t_sec:.6f}", "", "", "", "", "", "", ""])

            # desenhar trajetória
            for i in range(1, len(trail)):
                p0 = (int(trail[i-1][0]), int(trail[i-1][1]))
                p1 = (int(trail[i][0]), int(trail[i][1]))
                cv2.line(annotated, p0, p1, (255, 255, 0), 2)

            cv2.namedWindow("Ball detection (ESC)", cv2.WINDOW_NORMAL)
            cv2.imshow("Ball detection (ESC)", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] CSV guardado em: {OUT_CSV}")

if __name__ == "__main__":
    main()
