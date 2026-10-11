import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from data_prep.click_corners import OUT, load_saved
from pipeline.layout import ROUNDS, warped_photo
from pipeline.rectify import CANVAS, TILE_SIZE, tile_centers

LABELS = Path("data/tiles.json")
WINDOW = "label tiles"
BOARD_SIZE = 650
ZOOM_SIZE = 300
PANEL = 420
RESOURCE_KEYS = {ord("w"): "wood", ord("b"): "brick", ord("s"): "sheep", ord("g"): "wheat", ord("o"): "ore", ord("d"): "desert"}
EXPECTED_RESOURCES = {"wood": 4, "sheep": 4, "wheat": 4, "brick": 3, "ore": 3, "desert": 1}
EXPECTED_NUMBERS = {2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 1}
ESC, BACKSPACE = 27, 8


class QuitRequested(Exception):
    pass


def load_labels() -> dict[str, list[dict]]:
    return json.loads(LABELS.read_text()) if LABELS.exists() else {}


def save_labels(labels: dict[str, list[dict]]) -> None:
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    LABELS.write_text(json.dumps(labels, indent=1))


def check_round(tiles: list[dict]) -> list[str]:
    problems = []
    resources = Counter(t["resource"] for t in tiles)
    numbers = Counter(t["number"] for t in tiles if t["number"] is not None)
    for resource, n in EXPECTED_RESOURCES.items():
        if resources[resource] != n:
            problems.append(f"{resource}: {resources[resource]} (expected {n})")
    for number in sorted(set(EXPECTED_NUMBERS) | set(numbers)):
        if numbers[number] != EXPECTED_NUMBERS.get(number, 0):
            problems.append(f"number {number}: {numbers[number]} (expected {EXPECTED_NUMBERS.get(number, 0)})")
    for i, t in enumerate(tiles):
        if (t["resource"] == "desert") != (t["number"] is None):
            problems.append(f"hex {i}: desert must have no number, others must have one")
    return problems


def render(warped: np.ndarray, tiles: list[dict], current: int, typed: str, stage: str, round_name: str) -> np.ndarray:
    scale = BOARD_SIZE / CANVAS
    board = cv2.resize(warped, (BOARD_SIZE, BOARD_SIZE), interpolation=cv2.INTER_AREA)
    for i, (x, y) in enumerate(tile_centers()):
        px, py = int(x * scale), int(y * scale)
        label_y = py - 38
        if i == current:
            cv2.circle(board, (px, py), 30, (0, 255, 255), 3)
        cv2.circle(board, (px, label_y), 10, (0, 0, 0), -1)
        cv2.putText(board, str(i), (px - 8, label_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 255) if i == current else (255, 255, 255), 1)
        if i < len(tiles):
            t = tiles[i]
            text = t["resource"][:2] + ("" if t["number"] is None else str(t["number"]))
            cv2.putText(board, text, (px - 18, py + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(board, text, (px - 18, py + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    panel = np.full((BOARD_SIZE, PANEL, 3), 30, dtype=np.uint8)
    if current < 19:
        x, y = (int(v) for v in tile_centers()[current])
        r = int(TILE_SIZE * 1.1)
        zoom = warped[max(0, y - r):y + r, max(0, x - r):x + r]
        panel[10:10 + ZOOM_SIZE, 60:60 + ZOOM_SIZE] = cv2.resize(zoom, (ZOOM_SIZE, ZOOM_SIZE))
    lines = [
        f"round {round_name}   hex {min(current, 18)} / 18",
        "resource: w wood  b brick  s sheep",
        "          g wheat  o ore  d desert",
        "number: 2-9, then 1+0 / 1+1 / 1+2 for 10-12",
        "backspace = undo    esc = quit",
        f"now: {stage}  {typed}",
    ]
    for j, line in enumerate(lines):
        cv2.putText(panel, line, (12, ZOOM_SIZE + 45 + j * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return np.hstack([board, panel])


def label_round(round_name: str, warped: np.ndarray) -> list[dict] | None:
    tiles: list[dict] = []
    resource = None
    typed = ""
    while True:
        current = len(tiles)
        stage = "done, enter = save, backspace = undo" if current == 19 else ("number" if resource else "resource")
        shown = f"{resource or ''} {typed}"
        cv2.imshow(WINDOW, render(warped, tiles, current, shown, stage, round_name))
        key = cv2.waitKey(30) & 0xFF
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1 or key == ESC:
            raise QuitRequested
        if key == 255:
            continue

        if key == BACKSPACE:
            if typed:
                typed = ""
            elif resource:
                resource = None
            elif tiles:
                tiles.pop()
            continue

        if current == 19:
            if key in (13, 10):
                return tiles
            continue

        if resource is None:
            if key in RESOURCE_KEYS:
                resource = RESOURCE_KEYS[key]
                if resource == "desert":
                    tiles.append({"resource": "desert", "number": None})
                    resource = None
            continue

        if not chr(key).isdigit():
            continue
        if typed == "1":
            if chr(key) in "012":
                tiles.append({"resource": resource, "number": 10 + int(chr(key))})
                resource, typed = None, ""
            continue
        if chr(key) == "1":
            typed = "1"
        elif chr(key) in "2345689":
            tiles.append({"resource": resource, "number": int(chr(key))})
            resource = None


def main() -> None:
    rounds = json.loads(ROUNDS.read_text())
    corners = load_saved(OUT)
    labels = load_labels()
    todo = [name for name in rounds if name not in labels]
    if not todo:
        print(f"all rounds labeled in {LABELS}")
        return

    cv2.namedWindow(WINDOW)
    try:
        for name in todo:
            warped = warped_photo(rounds[name]["empty"], corners)
            while True:
                tiles = label_round(name, warped)
                problems = check_round(tiles)
                if not problems:
                    break
                print(f"round {name} doesn't match the Catan counts, relabel it:")
                for problem in problems:
                    print("   " + problem)
            labels[name] = tiles
            save_labels(labels)
            print(f"saved round {name}")
    except QuitRequested:
        print("quit")
    finally:
        cv2.destroyAllWindows()
    print(f"{len(labels)} of {len(rounds)} rounds labeled in {LABELS}")


if __name__ == "__main__":
    main()
