from pathlib import Path
import cv2

VIDEOS_DIR = Path("data/new_videos_raw")
OUT_DIR = Path("data/new_dataset/images")

# extrair 1 frame a cada N (ajustável)
EVERY_N_FRAMES = 3   # em vídeos curtos, isto dá muitos frames; podes subir para 3 ou 4

def extract(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERRO] Não consegui abrir: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[INFO] {video_path.name}: fps={fps:.1f}, frames={total}")

    i = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if i % EVERY_N_FRAMES == 0:
            out_name = f"{video_path.stem}_f{i:04d}.jpg"
            out_path = OUT_DIR / out_name
            cv2.imwrite(str(out_path), frame)
            saved += 1

        i += 1

    cap.release()
    return saved

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(VIDEOS_DIR.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError("Não encontrei .mp4 em data/videos_raw/")

    total_saved = 0
    for vp in videos:
        total_saved += extract(vp)

    print(f"[DONE] Frames guardados: {total_saved} em {OUT_DIR}")

if __name__ == "__main__":
    main()
