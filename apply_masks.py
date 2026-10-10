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
    ...


def to_pixels(box: tuple[float, float, float, float], w: int, h: int) -> tuple[int, int, int, int]:
    ...


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
