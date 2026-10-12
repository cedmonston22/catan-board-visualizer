import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from pipeline.rectify import compute_homography, ideal_corners

CORNER_WEIGHTS = Path("runs/detect/corners_v1/weights/best.pt")
CORNER_IMGSZ = 1024
CORNER_CONF = 0.03
MAX_CANDIDATES = 10
MAX_FIT_ERROR = 15.0


@dataclass
class CornerResult:
    points: list[list[float]] | None
    problem: str | None
    fit_error: float | None


def load_corner_model() -> YOLO | None:
    if not CORNER_WEIGHTS.exists():
        print(f"no corner model at {CORNER_WEIGHTS}, run train_corners.py first")
        return None
    return YOLO(str(CORNER_WEIGHTS))


def clockwise_from_top(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = [math.atan2(y - center[1], x - center[0]) for x, y in points]
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 1]))
    return np.roll(ordered, -start, axis=0)


def fit_error(points: np.ndarray) -> float:
    homography = compute_homography(points.tolist())
    mapped = cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), homography).reshape(-1, 2)
    return float(np.linalg.norm(mapped - ideal_corners(), axis=1).max())


def distinct_candidates(centers: np.ndarray, min_gap: float) -> np.ndarray:
    kept: list[np.ndarray] = []
    for c in centers:
        if all(np.linalg.norm(c - k) >= min_gap for k in kept):
            kept.append(c)
        if len(kept) == MAX_CANDIDATES:
            break
    return np.array(kept)


def best_hexagon(candidates: np.ndarray) -> tuple[np.ndarray | None, float]:
    best, best_error = None, float("inf")
    for combo in itertools.combinations(range(len(candidates)), 6):
        ordered = clockwise_from_top(candidates[list(combo)])
        if len(cv2.convexHull(ordered.astype(np.float32))) != 6:
            continue
        error = fit_error(ordered)
        if error < best_error:
            best, best_error = ordered, error
    return best, best_error


def find_corners(model: YOLO, image_path: Path) -> CornerResult:
    result = model.predict(str(image_path), imgsz=CORNER_IMGSZ, conf=CORNER_CONF, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    if len(boxes) < 6:
        return CornerResult(None, f"found only {len(boxes)} corners", None)

    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    sizes = (boxes[:, 2:] - boxes[:, :2]).mean(axis=1)
    candidates = distinct_candidates(centers, float(np.median(sizes)))
    if len(candidates) < 6:
        return CornerResult(None, f"found only {len(candidates)} distinct corners", None)

    ordered, error = best_hexagon(candidates)
    if ordered is None:
        return CornerResult(None, "no 6 corners form a convex hexagon", None)
    if error > MAX_FIT_ERROR:
        return CornerResult(ordered.tolist(), f"corners don't fit a Catan board (error {error:.1f} px)", error)
    return CornerResult(ordered.tolist(), None, error)
