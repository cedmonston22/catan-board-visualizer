import json
from pathlib import Path

import cv2
import numpy as np

from extract_crops import INDEX, OUTPUT
from label_colors import LABELS

COLORS = ["red", "blue", "white", "orange"]
REFS = Path("data/color_refs.json")
SMALL = 48
CENTER = 0.4
K = 3
MIN_CLUSTER = 0.1


def lab_pixels(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    scale = SMALL / max(h, w)
    small = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3)


def center_pixels(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    my = int(h * (1 - CENTER) / 2)
    mx = int(w * (1 - CENTER) / 2)
    return lab_pixels(crop[my:h - my, mx:w - mx])


def clusters(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, assignment, centers = cv2.kmeans(pixels, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    fractions = np.bincount(assignment.ravel(), minlength=K) / len(pixels)
    return centers, fractions


def reference_colors(files_by_color: dict[str, list[str]]) -> dict[str, np.ndarray]:
    refs = {}
    for color in files_by_color:
        crop_colors = []
        for file in files_by_color[color]:
            crop = cv2.imread(str(OUTPUT/file))
            pixels = center_pixels(crop)
            crop_colors.append(np.median(pixels, axis=0))
        crop_colors = np.median(np.array(crop_colors), axis=0)
        refs[color] = np.array(crop_colors)
    return refs
            
        


def classify(crop: np.ndarray, refs: dict[str, np.ndarray]) -> tuple[str, float]:
    crop_pixels = lab_pixels(crop)
    centers, fracs = clusters(crop_pixels)
    best_color = None
    best_dist = float("inf")
    
    for center, frac in zip(centers, fracs):
        if frac < MIN_CLUSTER:
            continue
        for color, ref in refs.items():
            distance = np.linalg.norm(center - ref)
            if distance < best_dist:
                best_dist = distance
                best_color = color
    return best_color, float(best_dist)


def evaluate(test: dict[str, str], refs: dict[str, np.ndarray]) -> None:
    confusion = {truth: {guess: 0 for guess in COLORS} for truth in COLORS}
    mistakes = []
    for name, truth in sorted(test.items()):
        guess, _ = classify(cv2.imread(str(OUTPUT / name)), refs)
        confusion[truth][guess] += 1
        if guess != truth:
            mistakes.append(f"{name}: {truth} -> {guess}")

    correct = sum(confusion[c][c] for c in COLORS)
    print(f"accuracy {correct}/{len(test)} = {correct / len(test):.1%}")
    print("truth \\ guess  " + "".join(f"{c:>8}" for c in COLORS))
    for truth in COLORS:
        print(f"{truth:>13}  " + "".join(f"{confusion[truth][g]:>8}" for g in COLORS))
    for line in mistakes:
        print("  " + line)


def main() -> None:
    if not LABELS.exists() or not INDEX.exists():
        print("run extract_crops.py and label_colors.py first")
        return
    labels = {name: color for name, color in json.loads(LABELS.read_text()).items() if color in COLORS}
    split_of = {row["file"]: row["split"] for row in json.loads(INDEX.read_text())}

    def group(names: list[str]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {color: [] for color in COLORS}
        for name in names:
            grouped[labels[name]].append(name)
        return grouped

    train = [name for name in labels if split_of.get(name) == "train"]
    test = {name: labels[name] for name in labels if split_of.get(name) == "valid"}
    print(f"{len(train)} train crops build the references, {len(test)} valid crops test them")

    refs = reference_colors(group(train))
    evaluate(test, refs)

    final = reference_colors(group(list(labels)))
    REFS.write_text(json.dumps({color: [round(float(v), 2) for v in ref] for color, ref in final.items()}, indent=2))
    print(f"references from all {len(labels)} labels saved to {REFS}")


if __name__ == "__main__":
    main()
