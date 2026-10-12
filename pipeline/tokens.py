import cv2
import numpy as np

MIN_RADIUS = 23
MAX_RADIUS = 32
DEFAULT_RADIUS = 28
MAX_OFFSET = 40
OUT_SIZE = 96
GRAY = (114, 114, 114)


def find_token(crop: np.ndarray, max_offset: int = MAX_OFFSET) -> tuple[int, int, int] | None:
    h, w = crop.shape[:2]
    gray = cv2.medianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=8, param1=80, param2=16,
                               minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)
    if circles is None:
        return None
    for x, y, r in circles[0]:
        if np.hypot(x - w / 2, y - h / 2) < max_offset:
            return int(x), int(y), int(r)
    return None


def isolate_token(crop: np.ndarray, max_offset: int = MAX_OFFSET) -> np.ndarray:
    h, w = crop.shape[:2]
    found = find_token(crop, max_offset)
    x, y, r = found if found else (w // 2, h // 2, DEFAULT_RADIUS)

    half = r + 2
    padded = cv2.copyMakeBorder(crop, half, half, half, half, cv2.BORDER_CONSTANT, value=GRAY)
    square = padded[y:y + 2 * half, x:x + 2 * half].copy()

    mask = np.zeros(square.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (half, half), r + 1, 255, -1)
    square[mask == 0] = GRAY
    return cv2.resize(square, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
