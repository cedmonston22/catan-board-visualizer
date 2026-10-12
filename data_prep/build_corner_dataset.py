import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from data_prep.click_corners import OUT, load_saved
from pipeline.layout import ROUNDS
from pipeline.rectify import find_image

DATASET = Path("datasets/corners")
VAL_ROUNDS = {"3"}
BOX_FRACTION = 0.25


def corner_boxes(points: np.ndarray, w: int, h: int) -> list[tuple[float, float, float, float]]:
    side = np.mean(np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)) * BOX_FRACTION
    return [(x / w, y / h, side / w, side / h) for x, y in points]


def main() -> None:
    rounds = json.loads(ROUNDS.read_text())
    corners = load_saved(OUT)
    if DATASET.exists():
        shutil.rmtree(DATASET)

    counts = {"train": 0, "val": 0}
    for name, info in rounds.items():
        split = "val" if name in VAL_ROUNDS else "train"
        (DATASET / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET / split / "labels").mkdir(parents=True, exist_ok=True)
        for pid in info["photos"]:
            image_path = find_image(pid)
            h, w = cv2.imread(str(image_path)).shape[:2]
            boxes = corner_boxes(np.array(corners[pid], dtype=np.float64), w, h)
            shutil.copy2(image_path, DATASET / split / "images" / f"{pid}.jpg")
            lines = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cx, cy, bw, bh in boxes]
            (DATASET / split / "labels" / f"{pid}.txt").write_text("\n".join(lines) + "\n")
            counts[split] += 1

    yaml_text = f"path: {DATASET.resolve().as_posix()}\ntrain: train/images\nval: val/images\nnc: 1\nnames: ['corner']\n"
    (DATASET / "data.yaml").write_text(yaml_text)
    print(f"{counts['train']} train photos, {counts['val']} val photos, 6 corners each, dataset at {DATASET}")


if __name__ == "__main__":
    main()
