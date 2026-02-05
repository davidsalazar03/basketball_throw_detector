import cv2
from ultralytics import YOLO

VIDEO_PATH = "data/videos/Basket_free_shots_1.mp4"
MODEL_PATH = "yolov8n.pt"   # já deve existir na pasta atual; senão, faz download

def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não consegui abrir o vídeo: {VIDEO_PATH}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # inferência (classe default COCO)
        results = model.predict(frame, conf=0.25, verbose=False)[0]

        # desenhar boxes
        annotated = results.plot()

        cv2.imshow("YOLO video test", annotated)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
