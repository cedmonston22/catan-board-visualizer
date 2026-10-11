import json
from pathlib import Path

import cv2
import numpy as np

IMAGES = [Path("datasets/catan/train/images"), Path("datasets/catan/valid/images"), Path("datasets/test_photos")]
ONLY = {"test01"}
OUT = Path("data/corners.json")
WINDOW = "click corners"
MAX_DISPLAY_HEIGHT = 650
NUM_POINTS = 6


def photo_id(image_path: Path) -> str:
    pid = image_path.name.split("_jpg")[0]
    return pid


def load_saved(out_path: Path) -> dict[str, list[list[float]]]:
    if out_path.exists():
        with open(out_path, "r") as file:
            return json.load(file)
    return {}


def save_all(out_path: Path, corners: dict[str, list[list[float]]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as file:
        json.dump(corners, file, indent=2)
        


def fit_to_screen(image: np.ndarray) -> tuple[np.ndarray, float]:
    scale = MAX_DISPLAY_HEIGHT / image.shape[0]
    scale = min(1.0, scale)
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return (resized, scale)


class QuitRequested(Exception):
    pass


def on_mouse(event: int, x: int, y: int, flags: int, state: dict) -> None:
    if event == cv2.EVENT_LBUTTONDOWN and len(state["points"]) < NUM_POINTS:
        state["points"].append((x, y))


def draw(display: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    canvas = display.copy()
    for i in range(1, len(points)):
        cv2.line(canvas, points[i - 1], points[i], (0, 255, 255), 1)
    if len(points) == NUM_POINTS:
        cv2.line(canvas, points[-1], points[0], (0, 255, 255), 1)
    for i, point in enumerate(points, start=1):
        cv2.circle(canvas, point, 5, (0, 255, 0), -1)
        cv2.putText(canvas, str(i), (point[0] + 7, point[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    status = f"{len(points)}/{NUM_POINTS}  u=undo  enter=save  s=skip  q=quit"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(canvas, status, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return canvas


def scale_back(points: list[tuple[int, int]], scale: float) -> list[list[float]]:
    res = []
    for x, y in points:
        new_x = x / scale
        new_y = y / scale
        res.append([new_x, new_y])
    return res


def click_image(image_path: Path) -> list[list[float]] | None:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"could not read {image_path}, skipping")
        return None
    display, scale = fit_to_screen(image)

    state = {"points": []}
    cv2.namedWindow(WINDOW)
    cv2.setWindowTitle(WINDOW, f"{photo_id(image_path)} - click {NUM_POINTS} corners clockwise from the top")
    cv2.setMouseCallback(WINDOW, on_mouse, state)

    while True:
        cv2.imshow(WINDOW, draw(display, state["points"]))
        key = cv2.waitKey(20) & 0xFF
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            raise QuitRequested
        if key == ord("u") and state["points"]:
            state["points"].pop()
        elif key in (13, 10) and len(state["points"]) == NUM_POINTS:
            return scale_back(state["points"], scale)
        elif key == ord("s"):
            return None
        elif key == ord("q"):
            raise QuitRequested


def main() -> None:
    corners = load_saved(OUT)
    all_images = [path for folder in IMAGES for path in sorted(folder.glob("*.jpg"))]
    todo = [
        path for path in all_images
        if photo_id(path) not in corners and (not ONLY or photo_id(path) in ONLY)
    ]
    print(f"{len(corners)} already saved, {len(todo)} to go")

    try:
        for image_path in todo:
            points = click_image(image_path)
            if points is None:
                print(f"skipped {photo_id(image_path)}")
                continue
            corners[photo_id(image_path)] = points
            save_all(OUT, corners)
            print(f"saved {photo_id(image_path)}")
    except QuitRequested:
        print("quit")
    finally:
        cv2.destroyAllWindows()
    print(f"{len(corners)} photos saved in {OUT}")


if __name__ == "__main__":
    main()
