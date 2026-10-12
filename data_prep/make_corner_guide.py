from pathlib import Path

import cv2
import numpy as np

from data_prep.click_corners import OUT, load_saved
from pipeline.rectify import find_image

PHOTO = "round03_01"
OUTPUT = Path("assets/corner_guide.jpg")
HEIGHT = 420
MARGIN = 0.06
ZOOM_CORNER = 1
ZOOM_HALF = 170
GAP = 10
DOT = (0, 230, 255)
OUTLINE = (0, 0, 0)
BACKGROUND = (24, 24, 24)


def draw_dot(image: np.ndarray, point: tuple[int, int], radius: int) -> None:
    cv2.circle(image, point, radius, DOT, -1, cv2.LINE_AA)
    cv2.circle(image, point, radius, OUTLINE, 2, cv2.LINE_AA)


def main() -> None:
    corners = load_saved(OUT).get(PHOTO)
    if corners is None:
        print(f"no clicked corners for {PHOTO} in {OUT}")
        return
    image = cv2.imread(str(find_image(PHOTO)))
    points = np.array(corners, dtype=np.float32)

    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    pad = MARGIN * max(x1 - x0, y1 - y0)
    left, top = int(max(0, x0 - pad)), int(max(0, y0 - pad))
    right, bottom = int(min(image.shape[1], x1 + pad)), int(min(image.shape[0], y1 + pad))
    scale = HEIGHT / (bottom - top)
    board = cv2.resize(image[top:bottom, left:right], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    shown = ((points - [left, top]) * scale).astype(np.int32)
    cv2.polylines(board, [shown], True, DOT, 2, cv2.LINE_AA)
    for x, y in shown:
        draw_dot(board, (int(x), int(y)), 8)
    cv2.circle(board, tuple(int(v) for v in shown[ZOOM_CORNER]), 16, (255, 255, 255), 2, cv2.LINE_AA)

    zx, zy = (int(v) for v in points[ZOOM_CORNER])
    zoom = image[max(0, zy - ZOOM_HALF):zy + ZOOM_HALF, max(0, zx - ZOOM_HALF):zx + ZOOM_HALF]
    zoom = cv2.resize(zoom, (HEIGHT, HEIGHT), interpolation=cv2.INTER_CUBIC)
    cv2.polylines(zoom, [((points - [zx - ZOOM_HALF, zy - ZOOM_HALF]) * HEIGHT / (2 * ZOOM_HALF)).astype(np.int32)],
                  True, DOT, 3, cv2.LINE_AA)
    draw_dot(zoom, (HEIGHT // 2, HEIGHT // 2), 14)
    cv2.rectangle(zoom, (0, 0), (HEIGHT - 1, HEIGHT - 1), (255, 255, 255), 3)

    guide = np.full((HEIGHT, board.shape[1] + GAP + HEIGHT, 3), BACKGROUND, dtype=np.uint8)
    guide[:, :board.shape[1]] = board
    guide[:, board.shape[1] + GAP:] = zoom

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT), guide, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"saved {OUTPUT} ({guide.shape[1]}x{guide.shape[0]}, {OUTPUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
