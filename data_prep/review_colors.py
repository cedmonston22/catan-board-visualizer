import json

import cv2
import numpy as np

from data_prep.extract_crops import OUTPUT
from data_prep.prelabel_colors import ALL_LABELS

WINDOW = "review colors"
COLS = 8
ROWS = 5
TILE = 110
HEADER = 44
CYCLE = ["red", "blue", "white", "orange", "unsure"]
BGR = {
    "red": (40, 40, 220),
    "blue": (200, 90, 30),
    "white": (245, 245, 245),
    "orange": (20, 140, 255),
    "unsure": (128, 128, 128),
}


def load() -> dict[str, dict]:
    return json.loads(ALL_LABELS.read_text())


def save(labels: dict[str, dict]) -> None:
    ALL_LABELS.write_text(json.dumps(labels, indent=1, sort_keys=True))


def make_pages(labels: dict[str, dict]) -> list[list[str]]:
    per_page = COLS * ROWS
    pages = []
    for color in CYCLE:
        group = sorted(name for name, v in labels.items() if not v["reviewed"] and v["color"] == color)
        for start in range(0, len(group), per_page):
            pages.append(group[start:start + per_page])
    return pages


def render(page: list[str], labels: dict[str, dict], thumbs: dict[str, np.ndarray], header: str) -> np.ndarray:
    canvas = np.full((HEADER + ROWS * TILE, COLS * TILE, 3), 30, dtype=np.uint8)
    cv2.putText(canvas, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, "click = change color   space = page is right, next   b = back   q = quit",
                (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1)
    for i, name in enumerate(page):
        r, c = divmod(i, COLS)
        x, y = c * TILE, HEADER + r * TILE
        canvas[y + 4:y + TILE - 4, x + 4:x + TILE - 4] = thumbs[name]
        color = labels[name]["color"]
        cv2.rectangle(canvas, (x + 2, y + 2), (x + TILE - 3, y + TILE - 3), BGR[color], 3)
        cv2.rectangle(canvas, (x + 4, y + 4), (x + 22, y + 20), (0, 0, 0), -1)
        cv2.putText(canvas, color[0].upper(), (x + 8, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BGR[color], 1)
    return canvas


def on_mouse(event: int, x: int, y: int, flags: int, state: dict) -> None:
    if event != cv2.EVENT_LBUTTONDOWN or y < HEADER:
        return
    i = ((y - HEADER) // TILE) * COLS + x // TILE
    page = state["pages"][state["index"]]
    if i < len(page):
        name = page[i]
        current = state["labels"][name]["color"]
        state["labels"][name]["color"] = CYCLE[(CYCLE.index(current) + 1) % len(CYCLE)]


def main() -> None:
    if not ALL_LABELS.exists():
        print(f"no {ALL_LABELS}, run prelabel_colors.py first")
        return
    labels = load()
    pages = make_pages(labels)
    if not pages:
        print("everything is already reviewed")
        return

    thumbs = {}
    for page in pages:
        for name in page:
            thumbs[name] = cv2.resize(cv2.imread(str(OUTPUT / name)), (TILE - 8, TILE - 8), interpolation=cv2.INTER_AREA)

    state = {"labels": labels, "pages": pages, "index": 0}
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse, state)

    while state["index"] < len(pages):
        page = pages[state["index"]]
        reviewed = sum(v["reviewed"] for v in labels.values())
        header = f"page {state['index'] + 1}/{len(pages)}   reviewed {reviewed}/{len(labels)}"
        cv2.imshow(WINDOW, render(page, labels, thumbs, header))
        key = cv2.waitKey(30) & 0xFF
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1 or key == ord("q"):
            break
        if key == ord(" "):
            for name in page:
                labels[name]["reviewed"] = True
            save(labels)
            state["index"] += 1
        elif key == ord("b") and state["index"] > 0:
            state["index"] -= 1

    save(labels)
    cv2.destroyAllWindows()
    reviewed = sum(v["reviewed"] for v in labels.values())
    print(f"reviewed {reviewed}/{len(labels)}, saved to {ALL_LABELS}")


if __name__ == "__main__":
    main()
