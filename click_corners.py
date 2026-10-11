import json
from pathlib import Path

import cv2
import numpy as np

IMAGES = Path("datasets/catan/train/images")
OUT = Path("data/corners.json")
WINDOW = "click corners"
MAX_DISPLAY_HEIGHT = 650
NUM_POINTS = 6


def photo_id(image_path: Path) -> str:
    pid = image_path.name.split("_jpg")[0]
    return pid


def load_saved(out_path: Path) -> dict[str, list[list[float]]]:
    if out_path.exists():
        with open(out_path, "r") as file:
            return json.load(file)
    return {}


def save_all(out_path: Path, corners: dict[str, list[list[float]]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as file:
        json.dump(corners, file, indent=2)
        


def fit_to_screen(image: np.ndarray) -> tuple[np.ndarray, float]:
    scale = MAX_DISPLAY_HEIGHT / image.shape[0]
    scale = min(1.0, scale)
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return (resized, scale)


def on_mouse(event: int, x: int, y: int, flags: int, state: dict) -> None:
    ...


def draw(display: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    ...


def click_image(image_path: Path) -> list[list[float]] | None:
    ...


def main() -> None:
    ...


if __name__ == "__main__":
    main()
