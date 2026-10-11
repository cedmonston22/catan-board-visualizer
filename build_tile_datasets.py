import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from apply_masks import DATASET as DETECT_DATASET
from apply_masks import load_names, read_labels
from click_corners import OUT, load_saved
from layout import ROTATIONS, ROUNDS, rotation_perm, warped_photo
from rectify import TILE_SIZE, compute_homography, find_image, tile_centers

TILE_LABELS = Path("data/tiles.json")
TILES_OUT = Path("datasets/tiles")
TOKENS_OUT = Path("datasets/tokens")
VAL_ROUNDS = {"3"}
HEX_HALF = int(TILE_SIZE * 0.9)
TOKEN_HALF = int(TILE_SIZE * 0.6)


def crop_square(image: np.ndarray, center: tuple[float, float], half: int) -> np.ndarray:
    x, y = int(center[0]), int(center[1])
    return image[max(0, y - half):y + half, max(0, x - half):x + half]


def robber_tile(pid: str, homography: np.ndarray, names: list[str]) -> int | None:
    image_path = find_image(pid)
    label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    if not label_path.exists():
        return None
    h, w = cv2.imread(str(image_path)).shape[:2]
    for cls, cx, cy, bw, bh in read_labels(label_path):
        if names[cls] == "robber":
            base = np.array([[[cx * w, (cy + bh * 0.25) * h]]], dtype=np.float32)
            point = cv2.perspectiveTransform(base, homography)[0, 0]
            return int(np.argmin(np.linalg.norm(np.array(tile_centers()) - point, axis=1)))
    return None


def photo_labels(round_labels: list[dict], rotation: int) -> list[dict]:
    inverse = np.argsort(rotation_perm(rotation))
    return [round_labels[int(inverse[j])] for j in range(19)]


def main() -> None:
    rounds = json.loads(ROUNDS.read_text())
    rotations = json.loads(ROTATIONS.read_text())
    tile_labels = json.loads(TILE_LABELS.read_text())
    corners = load_saved(OUT)
    names = load_names(DETECT_DATASET / "data.yaml")

    for out in (TILES_OUT, TOKENS_OUT):
        if out.exists():
            shutil.rmtree(out)

    counts = {"tiles": {}, "tokens": {}}
    skipped_tokens = 0
    for round_name, info in rounds.items():
        split = "val" if round_name in VAL_ROUNDS else "train"
        for pid in info["photos"]:
            warped = warped_photo(pid, corners)
            labels = photo_labels(tile_labels[round_name], rotations[pid]["rotation"])
            robber = robber_tile(pid, compute_homography(corners[pid]), names)

            for j, (center, label) in enumerate(zip(tile_centers(), labels)):
                name = f"{pid}_hex{j:02d}.jpg"
                resource = label["resource"]
                number = "none" if label["number"] is None else str(label["number"])

                folder = TILES_OUT / split / resource
                folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(folder / name), crop_square(warped, center, HEX_HALF))
                counts["tiles"][resource] = counts["tiles"].get(resource, 0) + 1

                if j == robber and number != "none":
                    skipped_tokens += 1
                    continue
                folder = TOKENS_OUT / split / number
                folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(folder / name), crop_square(warped, center, TOKEN_HALF))
                counts["tokens"][number] = counts["tokens"].get(number, 0) + 1

    for kind, by_class in counts.items():
        print(f"{kind:6} {sum(by_class.values()):5}  {dict(sorted(by_class.items()))}")
    print(f"skipped {skipped_tokens} token crops hidden under the robber")


if __name__ == "__main__":
    main()
