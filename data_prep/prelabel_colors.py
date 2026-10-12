import json
from pathlib import Path

from ultralytics import YOLO

from data_prep.extract_crops import INDEX, OUTPUT
from pipeline.snap import COLOR_IMGSZ, COLOR_WEIGHTS

ALL_LABELS = Path("data/colors_all.json")
MIN_OVERLAP = 0.7
BATCH = 256


def iou(a: list[int], b: list[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def carry_over(rows: list[dict], old: dict[str, dict]) -> tuple[dict[str, dict], list[dict]]:
    pool: dict[tuple[str, str], list[dict]] = {}
    for label in old.values():
        pool.setdefault((label["photo"], label["piece"]), []).append(label)

    labels, unmatched = {}, []
    for row in rows:
        candidates = pool.get((row["photo"], row["piece"]), [])
        best = max(candidates, key=lambda c: iou(row["box"], c["box"]), default=None)
        if best is not None and iou(row["box"], best["box"]) >= MIN_OVERLAP:
            candidates.remove(best)
            labels[row["file"]] = {**best, "box": row["box"]}
        else:
            unmatched.append(row)
    return labels, unmatched


def predict(rows: list[dict], model: YOLO) -> dict[str, dict]:
    labels = {}
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        results = model.predict([str(OUTPUT / r["file"]) for r in chunk], imgsz=COLOR_IMGSZ, verbose=False)
        for row, result in zip(chunk, results):
            labels[row["file"]] = {"color": result.names[result.probs.top1], "source": "auto", "reviewed": False,
                                   "photo": row["photo"], "piece": row["piece"], "box": row["box"]}
        print(f"pre-labeled {min(start + BATCH, len(rows))}/{len(rows)}")
    return labels


def main() -> None:
    if not INDEX.exists():
        print("run extract_crops.py first")
        return
    if not COLOR_WEIGHTS.exists():
        print(f"no color model at {COLOR_WEIGHTS}")
        return
    rows = json.loads(INDEX.read_text())
    old = json.loads(ALL_LABELS.read_text()) if ALL_LABELS.exists() else {}
    if any("box" not in label for label in old.values()):
        print(f"{ALL_LABELS} has labels without boxes; they can't be matched to new crops")
        return

    labels, unmatched = carry_over(rows, old)
    labels.update(predict(unmatched, YOLO(str(COLOR_WEIGHTS))))
    ALL_LABELS.write_text(json.dumps(labels, indent=1, sort_keys=True), encoding="utf-8")

    carried = len(rows) - len(unmatched)
    dropped = len(old) - carried
    print(f"{carried} labels carried over, {len(unmatched)} new crops pre-labeled with {COLOR_WEIGHTS.parent.parent.name}, "
          f"{dropped} old labels had no matching crop")
    print(f"saved to {ALL_LABELS}")


if __name__ == "__main__":
    main()
