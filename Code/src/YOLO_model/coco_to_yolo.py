import json
from pathlib import Path

COCO_TRAIN = Path(r"data/new_dataset/labels/train/train/_annotations.coco.json")
COCO_VAL   = Path(r"data/new_dataset/labels/val/valid/_annotations.coco.json")

IMG_TRAIN_DIR = Path(r"data/new_dataset/labels/train/train")
IMG_VAL_DIR   = Path(r"data/new_dataset/labels/val/valid")

LAB_TRAIN_DIR = Path(r"data/new_ball_yolo/labels/train")
LAB_VAL_DIR   = Path(r"data/new_ball_yolo/labels/val")
# No teu COCO:
TARGET_CLASS_NAME = "basketball"   # vamos mapear para YOLO class 0

def convert(coco_json: Path, img_dir: Path, lab_dir: Path):
    data = json.loads(coco_json.read_text(encoding="utf-8"))

    # encontrar todos os category_id com o nome 'basketballs'
    target_ids = {c["id"] for c in data["categories"]
                  if c["name"].strip().lower() == TARGET_CLASS_NAME}

    if not target_ids:
        names = [c["name"] for c in data["categories"]]
        raise RuntimeError(f"Não encontrei categoria '{TARGET_CLASS_NAME}'. Categorias: {names}")

    images = {im["id"]: im for im in data["images"]}

    ann_by_img = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        if ann["category_id"] not in target_ids:
            continue
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    lab_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    missing = 0

    for img_id, im in images.items():
        file_name = Path(im["file_name"]).name
        w = float(im["width"])
        h = float(im["height"])

        img_path = img_dir / file_name
        if not img_path.exists():
            print(f"[WARN] Imagem em falta: {img_path}")
            missing += 1
            continue

        lines = []
        for ann in ann_by_img.get(img_id, []):
            x, y, bw, bh = [float(v) for v in ann["bbox"]]


            xc = (x + bw/2.0) / w
            yc = (y + bh/2.0) / h
            wn = bw / w
            hn = bh / h

            # clamp
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            wn = min(max(wn, 0.0), 1.0)
            hn = min(max(hn, 0.0), 1.0)

            # classe única -> 0
            lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        out_txt = lab_dir / (Path(file_name).stem + ".txt")
        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1

    return written, missing

def main():
    ntr, mistr = convert(COCO_TRAIN, IMG_TRAIN_DIR, LAB_TRAIN_DIR)
    nva, misva = convert(COCO_VAL, IMG_VAL_DIR, LAB_VAL_DIR)

    print(f"[DONE] train labels escritos: {ntr} | imgs em falta: {mistr}")
    print(f"[DONE] val   labels escritos: {nva} | imgs em falta: {misva}")

if __name__ == "__main__":
    main()
