import json
from pathlib import Path

import cv2
import numpy as np

from data_prep.color import COLORS, REFS, classify
from data_prep.extract_crops import INDEX, OUTPUT
from data_prep.label_colors import LABELS

ALL_LABELS = Path("data/colors_all.json")


def main() -> None:
    if not INDEX.exists() or not REFS.exists():
        print("run extract_crops.py and color.py first")
        return
    if ALL_LABELS.exists():
        print(f"{ALL_LABELS} already exists, delete it first to pre-label from scratch")
        return

    rows = json.loads(INDEX.read_text())
    human = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    refs = {color: np.array(ref, dtype=np.float32) for color, ref in json.loads(REFS.read_text()).items()}

    labels = {}
    for i, row in enumerate(rows, start=1):
        name = row["file"]
        if name in human:
            labels[name] = {"color": human[name], "source": "human", "reviewed": True}
        else:
            guess, _ = classify(cv2.imread(str(OUTPUT / name)), refs)
            labels[name] = {"color": guess, "source": "auto", "reviewed": False}
        if i % 250 == 0:
            print(f"{i}/{len(rows)}")

    ALL_LABELS.write_text(json.dumps(labels, indent=1, sort_keys=True))
    counts = {color: sum(v["color"] == color for v in labels.values()) for color in COLORS + ["unsure"]}
    auto = sum(v["source"] == "auto" for v in labels.values())
    print(f"{len(labels)} crops labeled ({auto} auto, {len(labels) - auto} from your hand labels): {counts}")
    print(f"saved to {ALL_LABELS}")


if __name__ == "__main__":
    main()
