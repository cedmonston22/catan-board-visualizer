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


def fraction_inside(box, region) -> float:
    ...


def process_split(split_dir: Path, ignore_id: int, remap: dict[int, int]) -> None:
    for label_path in sorted((split_dir / "labels").glob("*.txt")):
        ...


def write_yaml(yaml_path: Path) -> None:
    ...


def main() -> None:
    ...


if __name__ == "__main__":
    main()
