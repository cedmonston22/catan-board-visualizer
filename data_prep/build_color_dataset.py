import json
import shutil
from pathlib import Path

from data_prep.color import COLORS
from data_prep.extract_crops import INDEX, OUTPUT
from data_prep.prelabel_colors import ALL_LABELS

DATASET = Path("datasets/colors")
SPLIT_FOLDER = {"train": "train", "valid": "val"}


def main() -> None:
    if not ALL_LABELS.exists() or not INDEX.exists():
        print("run extract_crops.py, prelabel_colors.py and review_colors.py first")
        return
    labels = json.loads(ALL_LABELS.read_text())
    split_of = {row["file"]: row["split"] for row in json.loads(INDEX.read_text())}

    unreviewed = sum(not v["reviewed"] for v in labels.values())
    if unreviewed:
        print(f"{unreviewed} crops are not reviewed yet, finish review_colors.py first")
        return

    if DATASET.exists():
        shutil.rmtree(DATASET)
    for folder in SPLIT_FOLDER.values():
        for color in COLORS:
            (DATASET / folder / color).mkdir(parents=True)

    counts = {folder: {color: 0 for color in COLORS} for folder in SPLIT_FOLDER.values()}
    skipped = 0
    for name, label in labels.items():
        color = label["color"]
        if color not in COLORS or name not in split_of:
            skipped += 1
            continue
        folder = SPLIT_FOLDER[split_of[name]]
        shutil.copy2(OUTPUT / name, DATASET / folder / color / name)
        counts[folder][color] += 1

    for folder, by_color in counts.items():
        print(f"{folder:5} {sum(by_color.values()):5}  {by_color}")
    print(f"skipped {skipped}, dataset at {DATASET}")


if __name__ == "__main__":
    main()
