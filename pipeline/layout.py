import json
import math
from pathlib import Path

import cv2
import numpy as np

from data_prep.click_corners import OUT, load_saved
from pipeline.rectify import CENTER, TILE_SIZE, compute_homography, find_image, tile_centers, warp

ROUNDS = Path("data/rounds.json")
ROTATIONS = Path("outputs/rotations.json")
PATCH = 0.55


def rotation_perm(k: int) -> np.ndarray:
    centers = np.array(tile_centers())
    angle = math.radians(60 * k)
    cos, sin = math.cos(angle), math.sin(angle)
    perm = []
    for x, y in centers:
        dx, dy = x - CENTER[0], y - CENTER[1]
        rotated = np.array([CENTER[0] + dx * cos - dy * sin, CENTER[1] + dx * sin + dy * cos])
        perm.append(int(np.argmin(np.linalg.norm(centers - rotated, axis=1))))
    return np.array(perm)


def tile_features(warped: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(warped.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    r = int(TILE_SIZE * PATCH)
    colors = []
    for x, y in tile_centers():
        x, y = int(x), int(y)
        patch = lab[y - r:y + r, x - r:x + r].reshape(-1, 3)
        colors.append(np.median(patch, axis=0))
    colors = np.array(colors)
    return (colors - colors.mean(axis=0)) / (colors.std(axis=0) + 1e-6)


def best_rotation(photo: np.ndarray, empty: np.ndarray, perms: list[np.ndarray]) -> tuple[int, float]:
    best_k, best_dist = 0, float("inf")
    for k, perm in enumerate(perms):
        dist = float(np.median(np.linalg.norm(photo[perm] - empty, axis=1)))
        if dist < best_dist:
            best_k, best_dist = k, dist
    return best_k, best_dist


def warped_photo(pid: str, corners: dict[str, list[list[float]]]) -> np.ndarray:
    return warp(cv2.imread(str(find_image(pid))), compute_homography(corners[pid]))


def main() -> None:
    rounds = json.loads(ROUNDS.read_text())
    corners = load_saved(OUT)
    perms = [rotation_perm(k) for k in range(6)]

    rotations = {}
    for name, info in rounds.items():
        empty = tile_features(warped_photo(info["empty"], corners))
        for pid in info["photos"]:
            k, dist = best_rotation(tile_features(warped_photo(pid, corners)), empty, perms)
            rotations[pid] = {"round": name, "rotation": k, "distance": round(dist, 3)}
            print(f"round {name}  {pid:12} rotation {k}  distance {dist:.2f}")

    ROTATIONS.parent.mkdir(parents=True, exist_ok=True)
    ROTATIONS.write_text(json.dumps(rotations, indent=2))
    print(f"saved {len(rotations)} rotations to {ROTATIONS}")


if __name__ == "__main__":
    main()
