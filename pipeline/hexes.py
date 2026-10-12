from dataclasses import dataclass
from pathlib import Path
from collections import Counter

import numpy as np
from ultralytics import YOLO

from pipeline.rectify import TILE_SIZE, tile_centers
from pipeline.tokens import find_token, isolate_token

TILE_WEIGHTS = Path("runs/classify/tiles_v2/weights/best.pt")
TOKEN_WEIGHTS = Path("runs/classify/tokens_v2/weights/best.pt")
TILE_IMGSZ = 224
TOKEN_IMGSZ = 96
HEX_HALF = int(TILE_SIZE * 0.9)
TOKEN_HALF = int(TILE_SIZE * 0.6)
TOKEN_SEARCH_OFFSET = 70
EXPECTED_COUNTS = {2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 1}

@dataclass
class Hex:
    resource: str
    number: int | None
    resource_conf: float
    number_conf: float


def crop_square(image: np.ndarray, center: tuple[float, float], half: int) -> np.ndarray:
    x, y = int(center[0]), int(center[1])
    return image[max(0, y - half):y + half, max(0, x - half):x + half]


def load_hex_models() -> tuple[YOLO, YOLO] | None:
    for weights in (TILE_WEIGHTS, TOKEN_WEIGHTS):
        if not weights.exists():
            print(f"no classifier at {weights}, run its training script first")
            return None
    return YOLO(str(TILE_WEIGHTS)), YOLO(str(TOKEN_WEIGHTS))


def read_hexes(warped: np.ndarray, tile_model: YOLO, token_model: YOLO) -> list[Hex]:
    centers = tile_centers()
    hex_crops = [crop_square(warped, c, HEX_HALF) for c in centers]
    token_found = [find_token(crop, TOKEN_SEARCH_OFFSET) is not None for crop in hex_crops]
    tile_results = tile_model.predict(hex_crops, imgsz=TILE_IMGSZ, verbose=False)
    token_crops = [isolate_token(crop, TOKEN_SEARCH_OFFSET) for crop in hex_crops]
    token_results = token_model.predict(token_crops, imgsz=TOKEN_IMGSZ, verbose=False)

    hexes = []
    for tile, token, found in zip(tile_results, token_results, token_found):
        resource = tile.names[tile.probs.top1]
        label = token.names[token.probs.top1]
        number = None if resource == "desert" or label == "none" or not found else int(label)
        hexes.append(Hex(resource, number, float(tile.probs.top1conf), float(token.probs.top1conf)))
    return hexes

def fill_unknown_numbers(hexes: list[Hex]) -> None:
    hex_nums = [h.number for h in hexes if h.number is not None]
    known_nums = Counter(hex_nums)
    not_assigned = []
    for num, count in EXPECTED_COUNTS.items():
        temp = count - known_nums[num]
        if temp > 0:
            not_assigned.extend([num] * temp)
        elif temp < 0:
            return
    unknown_tiles = [h for h in hexes if h.number is None and h.resource != "desert"]
    if len(not_assigned) == 1 and len(unknown_tiles) == 1:
        unknown_tiles[0].number = not_assigned[0]
        