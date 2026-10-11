import math
from pathlib import Path

import cv2
import numpy as np

from click_corners import IMAGES, OUT, load_saved

TILE_SIZE = 100
CANVAS = 1000
CENTER = (CANVAS / 2, CANVAS / 2)
CORNER_RADIUS = 2.5 * math.sqrt(3) * TILE_SIZE
OUTPUT = Path("outputs/rectified")


def ideal_corners() -> np.ndarray:
    points = []
    for i in range(6):
        angle = math.radians(90 - 60 * i)
        x = CENTER[0] + CORNER_RADIUS * math.cos(angle)
        y = CENTER[1] - CORNER_RADIUS * math.sin(angle)
        points.append((x, y))
    return np.array(points, dtype=np.float32)


def find_image(pid: str) -> Path:
    for folder in IMAGES:
        matches = sorted(folder.glob(f"{pid}_jpg*.jpg"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no image found for {pid} in {IMAGES}")


def compute_homography(clicked: list[list[float]]) -> np.ndarray:
    src = np.array(clicked, dtype=np.float32)
    dst = ideal_corners()
    homography, _ = cv2.findHomography(src, dst)
    if homography is None:
        raise ValueError("could not compute a homography from these corners")
    return homography


def warp(image: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(image, homography, (CANVAS, CANVAS))


def tile_centers() -> list[tuple[float, float]]:
    centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) > 2:
                continue
            x = CENTER[0] + TILE_SIZE * 1.5 * q
            y = CENTER[1] + TILE_SIZE * math.sqrt(3) * (r + q / 2)
            centers.append((x, y))
    return centers


def hex_outline(center: tuple[float, float]) -> np.ndarray:
    points = []
    for k in range(6):
        angle = math.radians(60 * k)
        points.append((center[0] + TILE_SIZE * math.cos(angle), center[1] + TILE_SIZE * math.sin(angle)))
    return np.array(points, dtype=np.int32)


def draw_grid(warped: np.ndarray) -> np.ndarray:
    canvas = warped.copy()
    for center in tile_centers():
        cv2.polylines(canvas, [hex_outline(center)], True, (0, 255, 255), 2)
        cv2.circle(canvas, (int(center[0]), int(center[1])), 4, (0, 255, 255), -1)
    for x, y in ideal_corners():
        cv2.circle(canvas, (int(x), int(y)), 7, (0, 255, 0), -1)
    return canvas


def main() -> None:
    corners = load_saved(OUT)
    if not corners:
        print(f"no corners in {OUT}, run click_corners.py first")
        return

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for pid, clicked in corners.items():
        image = cv2.imread(str(find_image(pid)))
        warped = warp(image, compute_homography(clicked))
        cv2.imwrite(str(OUTPUT / f"{pid}.jpg"), draw_grid(warped))
        print(f"rectified {pid}")
    print(f"{len(corners)} images saved to {OUTPUT}")


if __name__ == "__main__":
    main()
