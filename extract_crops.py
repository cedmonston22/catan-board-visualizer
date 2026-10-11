import json
from pathlib import Path

import cv2
import numpy as np

from apply_masks import DATASET, load_names, read_labels, to_pixels
from click_corners import photo_id

SPLITS = ["train", "valid"]
SKIP = {"robber"}
PAD = 0.1
OUTPUT = Path("outputs/crops")
INDEX = OUTPUT / "index.json"


def crop_box(image: np.ndarray, box: tuple[int, int, int, int], pad: float) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = int((x2 - x1) * pad)
    pad_y = int((y2 - y1) * pad)
    x1, x2 = max(0, x1 - pad_x), min(w, x2 + pad_x)
    y1, y2 = max(0, y1 - pad_y), min(h, y2 + pad_y)
    return image[y1:y2, x1:x2]


def extract_split(split: str, names: list[str]) -> list[dict]:
    rows = []
    split_dir = DATASET / split
    for label_path in sorted((split_dir / "labels").glob("*.txt")):
        labels = read_labels(label_path)
        if not labels:
            continue
        image_path = next((split_dir / "images").glob(f"{label_path.stem}.*"))
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        pid = photo_id(image_path)

        for line, label in enumerate(labels):
            piece = names[label[0]]
            if piece in SKIP:
                continue
            box = to_pixels(label[1:], w, h)
            crop = crop_box(image, box, PAD)
            if crop.size == 0:
                continue
            name = f"{pid}_{line:03d}_{piece}.jpg"
            cv2.imwrite(str(OUTPUT / name), crop)
            rows.append({"file": name, "photo": pid, "piece": piece, "split": split, "box": list(box)})
    return rows


def main() -> None:
    names = load_names(DATASET / "data.yaml")
    if "ignore" in names:
        print("data.yaml still has 'ignore', run apply_masks.py first")
        return

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in SPLITS:
        rows.extend(extract_split(split, names))
    INDEX.write_text(json.dumps(rows, indent=2))

    counts = {}
    for row in rows:
        counts[row["piece"]] = counts.get(row["piece"], 0) + 1
    print(f"{len(rows)} crops saved to {OUTPUT}: {counts}")


if __name__ == "__main__":
    main()
