import itertools
import math
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from pipeline.hexes import HEX_HALF, TILE_IMGSZ, crop_square
from pipeline.rectify import compute_homography, ideal_corners, tile_centers, warp
from pipeline.weights import weights_path

CORNER_WEIGHTS = weights_path("corners")
CORNER_IMGSZ = 1024
CORNER_CONF = 0.03
MAX_CANDIDATES = 10
MAX_FIT_ERROR = 40.0
MAX_TRIES = 25
MIN_TILE_CONF = 0.8
EXPECTED_RESOURCES = {"wood": 4, "sheep": 4, "wheat": 4, "brick": 3, "ore": 3, "desert": 1}


@dataclass
class CornerResult:
    points: list[list[float]] | None
    problem: str | None
    tile_conf: float | None


def load_corner_model() -> YOLO | None:
    if not CORNER_WEIGHTS.exists():
        print(f"no corner model at {CORNER_WEIGHTS}, promote one with python -m pipeline.promote corners path/to/best.pt")
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


def is_convex_hexagon(points: np.ndarray) -> bool:
    return len(cv2.convexHull(points.astype(np.float32))) == 6


def distinct_candidates(centers: np.ndarray, confs: np.ndarray, min_gap: float) -> tuple[np.ndarray, np.ndarray]:
    kept, kept_conf = [], []
    for c, conf in zip(centers, confs):
        if all(np.linalg.norm(c - k) >= min_gap for k in kept):
            kept.append(c)
            kept_conf.append(conf)
        if len(kept) == MAX_CANDIDATES:
            break
    return np.array(kept), np.array(kept_conf)


def hexagons_from_six(candidates: np.ndarray) -> list[tuple[float, np.ndarray]]:
    found = []
    for combo in itertools.combinations(range(len(candidates)), 6):
        ordered = clockwise_from_top(candidates[list(combo)])
        if is_convex_hexagon(ordered):
            error = fit_error(ordered)
            if error <= MAX_FIT_ERROR:
                found.append((error, ordered))
    return sorted(found, key=lambda f: f[0])


def hexagons_from_five(candidates: np.ndarray, confs: np.ndarray, w: int, h: int) -> list[tuple[float, np.ndarray]]:
    ideal = ideal_corners()
    found = []
    for combo in itertools.combinations(range(len(candidates)), 5):
        five = clockwise_from_top(candidates[list(combo)]).astype(np.float32)
        for missing in range(6):
            slots = [(missing + 1 + i) % 6 for i in range(5)]
            homography, _ = cv2.findHomography(five, ideal[slots])
            if homography is None:
                continue
            sixth = cv2.perspectiveTransform(ideal[[missing]].reshape(-1, 1, 2), np.linalg.inv(homography)).reshape(2)
            if not (-0.2 * w <= sixth[0] <= 1.2 * w and -0.2 * h <= sixth[1] <= 1.2 * h):
                continue
            ordered = clockwise_from_top(np.vstack([five, sixth]))
            if is_convex_hexagon(ordered):
                found.append((-float(confs[list(combo)].sum()), ordered))
    return sorted(found, key=lambda f: f[0])


def tile_check(image: np.ndarray, points: np.ndarray, tile_model: YOLO) -> float | None:
    warped = warp(image, compute_homography(points.tolist()))
    crops = [crop_square(warped, c, HEX_HALF) for c in tile_centers()]
    results = tile_model.predict(crops, imgsz=TILE_IMGSZ, verbose=False)
    resources = Counter(r.names[r.probs.top1] for r in results)
    mean_conf = float(np.mean([float(r.probs.top1conf) for r in results]))
    if dict(resources) != EXPECTED_RESOURCES or mean_conf < MIN_TILE_CONF:
        return None
    return mean_conf


def find_corners(model: YOLO, tile_model: YOLO, image: np.ndarray) -> CornerResult:
    result = model.predict(image, imgsz=CORNER_IMGSZ, conf=CORNER_CONF, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    if len(boxes) < 5:
        return CornerResult(None, f"found only {len(boxes)} corners", None)

    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    sizes = (boxes[:, 2:] - boxes[:, :2]).mean(axis=1)
    candidates, confs = distinct_candidates(centers, result.boxes.conf.cpu().numpy(), float(np.median(sizes)))
    if len(candidates) < 5:
        return CornerResult(None, f"found only {len(candidates)} distinct corners", None)

    h, w = image.shape[:2]
    options = hexagons_from_six(candidates) if len(candidates) >= 6 else []
    options += hexagons_from_five(candidates, confs, w, h)
    for _, ordered in options[:MAX_TRIES]:
        tile_conf = tile_check(image, ordered, tile_model)
        if tile_conf is not None:
            return CornerResult(ordered.tolist(), None, tile_conf)
    return CornerResult(None, "no candidate corners produced a valid Catan board", None)
