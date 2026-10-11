import json
import random
from pathlib import Path

import cv2
import numpy as np

from extract_crops import INDEX, OUTPUT

LABELS = Path("data/colors.json")
SAMPLE_SIZE = 200
SEED = 0
WINDOW = "label colors"
DISPLAY_SIZE = 320
KEYS = {ord("r"): "red", ord("b"): "blue", ord("w"): "white", ord("o"): "orange", ord("x"): "unsure"}


def load_labels() -> dict[str, str]:
    if LABELS.exists():
        return json.loads(LABELS.read_text())
    return {}


def save_labels(labels: dict[str, str]) -> None:
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    LABELS.write_text(json.dumps(labels, indent=2, sort_keys=True))


def pick_sample(rows: list[dict]) -> list[str]:
    by_piece: dict[str, list[str]] = {}
    for row in rows:
        by_piece.setdefault(row["piece"], []).append(row["file"])
    rng = random.Random(SEED)
    per_piece = SAMPLE_SIZE // len(by_piece)
    sample = []
    for files in by_piece.values():
        sample.extend(rng.sample(files, min(per_piece, len(files))))
    rng.shuffle(sample)
    return sample


def render(crop: np.ndarray, done: int, total: int, name: str) -> np.ndarray:
    h, w = crop.shape[:2]
    scale = DISPLAY_SIZE / max(h, w)
    big = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((DISPLAY_SIZE + 60, max(DISPLAY_SIZE, 420), 3), 30, dtype=np.uint8)
    canvas[:big.shape[0], :big.shape[1]] = big
    cv2.putText(canvas, f"{done}/{total}  {name}", (6, DISPLAY_SIZE + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(canvas, "r b w o = color   x = unsure   u = undo   q = quit", (6, DISPLAY_SIZE + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return canvas


def main() -> None:
    if not INDEX.exists():
        print(f"no {INDEX}, run extract_crops.py first")
        return
    rows = json.loads(INDEX.read_text())
    labels = load_labels()
    sample = pick_sample(rows)
    todo = [name for name in sample if name not in labels]
    print(f"{len(labels)} labeled so far, {len(todo)} left in this sample")

    history: list[str] = []
    i = 0
    cv2.namedWindow(WINDOW)
    while i < len(todo):
        name = todo[i]
        crop = cv2.imread(str(OUTPUT / name))
        cv2.imshow(WINDOW, render(crop, len(labels), len(labels) + len(todo) - i, name))
        key = cv2.waitKey(0) & 0xFF
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1 or key == ord("q"):
            break
        if key == ord("u") and history:
            labels.pop(history.pop(), None)
            save_labels(labels)
            i -= 1
        elif key in KEYS:
            labels[name] = KEYS[key]
            history.append(name)
            save_labels(labels)
            i += 1

    cv2.destroyAllWindows()
    counts: dict[str, int] = {}
    for color in labels.values():
        counts[color] = counts.get(color, 0) + 1
    print(f"{len(labels)} labels saved to {LABELS}: {counts}")


if __name__ == "__main__":
    main()
