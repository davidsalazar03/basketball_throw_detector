from pathlib import Path
import shutil
import re

SRC = Path("data/new_dataset/images")  # onde estão os jpg agora
DST = Path("data/new_dataset")         # raiz do dataset YOLO

def main():
    imgs = sorted([p for p in SRC.glob("*.jpg") if p.is_file()])
    if not imgs:
        raise FileNotFoundError("Não há .jpg em data/new_dataset/images/")

    # agrupar por prefixo do vídeo: v1_f0000.jpg -> v1
    pat = re.compile(r"^(.*)_f\d+\.jpg$", re.IGNORECASE)
    groups = {}
    for p in imgs:
        m = pat.match(p.name)
        if not m:
            continue
        vid = m.group(1)
        groups.setdefault(vid, []).append(p)

    vids = sorted(groups.keys())
    print("Total vídeos encontrados:", len(vids))
    if len(vids) < 2:
        raise RuntimeError("Preciso de pelo menos 2 vídeos para split train/val por vídeo.")

    val_vid = vids[-3:]          # último para validação
    train_vids = vids[:-3]      # resto para treino

    print("[INFO] Train videos:", train_vids, "( total:", len(train_vids), ")")
    print("[INFO] Val videos:", val_vid, "( total:", len(val_vid), ")")

    # criar pastas
    (DST / "images" / "train").mkdir(parents=True, exist_ok=True)
    (DST / "images" / "val").mkdir(parents=True, exist_ok=True)
    (DST / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (DST / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # mover (ou copiar) imagens
    def put(p: Path, split: str):
        out = DST / "images" / split / p.name
        shutil.copy2(p, out)

    n_train = 0
    n_val = 0
    for vid, files in groups.items():
        if vid in val_vid:
            for p in files:
                put(p, "val")
                n_val += 1
        else:
            for p in files:
                put(p, "train")
                n_train += 1

    print(f"[DONE] train={n_train} val={n_val}")
    print("[NEXT] Agora anota as labels em data/dataset/labels/train e val com a classe 'ball' (id 0).")

if __name__ == "__main__":
    main()
