from pathlib import Path

import cv2
import yaml

DATASET = Path("datasets/catan")
KEEP = ["city", "road", "robber", "settlement"]
IGNORE = "ignore"
GRAY = (114, 114, 114)
INSIDE_THRESHOLD = 0.5


def load_names(yaml_path: Path) -> list[str]:
    with open(yaml_path, "r") as file:
        data_yaml = yaml.safe_load(file)
    return data_yaml["names"]


def build_remap(names: list[str]) -> dict[int, int]:
    remap = {}
    for new_id, name in enumerate(KEEP):
        remap[names.index(name)] = new_id
    return remap


def read_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    labels = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls, cx, cy, w, h = parts
        labels.append((int(cls), float(cx), float(cy), float(w), float(h)))
    return labels


def to_pixels(box: tuple[float, float, float, float], w: int, h: int) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = box
    left_edge = max(0, min(int((cx - bw / 2) * w), w))
    right_edge = max(0, min(int((cx + bw / 2) * w), w))
    top_edge = max(0, min(int((cy - bh / 2) * h), h))
    bottom_edge = max(0, min(int((cy + bh / 2) * h), h))
    return left_edge, top_edge, right_edge, bottom_edge


def fraction_inside(box: tuple[int, int, int, int], region: tuple[int, int, int, int]) -> float:
    bx1, by1, bx2, by2 = box
    rx1, ry1, rx2, ry2 = region
    overlap_w = max(0, min(bx2, rx2) - max(bx1, rx1))
    overlap_h = max(0, min(by2, ry2) - max(by1, ry1))
    box_area = (bx2 - bx1) * (by2 - by1)
    if box_area == 0:
        return 0.0
    return overlap_w * overlap_h / box_area


def process_split(split_dir: Path, ignore_id: int, remap: dict[int, int]) -> tuple[int, int]:
    masked_images = 0
    dropped_boxes = 0
    for label_path in sorted((split_dir / "labels").glob("*.txt")):
        labels = read_labels(label_path)
        ignores = [label for label in labels if label[0] == ignore_id]
        others = [label for label in labels if label[0] != ignore_id]

        if ignores:
            image_path = next((split_dir / "images").glob(f"{label_path.stem}.*"))
            image = cv2.imread(str(image_path))
            h, w = image.shape[:2]
            regions = [to_pixels(label[1:], w, h) for label in ignores]
            for x1, y1, x2, y2 in regions:
                cv2.rectangle(image, (x1, y1), (x2, y2), GRAY, -1)
            cv2.imwrite(str(image_path), image)
            masked_images += 1

            kept = []
            for label in others:
                box = to_pixels(label[1:], w, h)
                if any(fraction_inside(box, region) > INSIDE_THRESHOLD for region in regions):
                    dropped_boxes += 1
                else:
                    kept.append(label)
            others = kept

        lines = [f"{remap[cls]} {cx} {cy} {bw} {bh}" for cls, cx, cy, bw, bh in others]
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return masked_images, dropped_boxes


def write_yaml(yaml_path: Path) -> None:
    with open(yaml_path, "r") as file:
        data_yaml = yaml.safe_load(file)
    new_yaml = {
        "path": data_yaml["path"],
        "train": data_yaml["train"],
        "val": data_yaml["val"],
        "nc": len(KEEP),
        "names": KEEP,
    }
    with open(yaml_path, "w") as file:
        yaml.safe_dump(new_yaml, file, sort_keys=False)


def main() -> None:
    yaml_path = DATASET / "data.yaml"
    names = load_names(yaml_path)
    if IGNORE not in names:
        print(f"'{IGNORE}' not in {names}: dataset already processed, or wrong export. Nothing changed.")
        return

    ignore_id = names.index(IGNORE)
    remap = build_remap(names)
    for split in ("train", "valid"):
        masked, dropped = process_split(DATASET / split, ignore_id, remap)
        print(f"{split}: masked {masked} images, dropped {dropped} boxes under masks")
    write_yaml(yaml_path)
    print(f"data.yaml now has nc={len(KEEP)} names={KEEP}")


if __name__ == "__main__":
    main()
