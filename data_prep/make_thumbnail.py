from pathlib import Path

import cv2
import numpy as np

from data_prep.click_corners import OUT, load_saved
from pipeline.board import build_board
from pipeline.read_board import load_models, read_board
from pipeline.rectify import CANVAS, find_image
from pipeline.render import SEA_BGR

PHOTO = "round03_01"
OUTPUT = Path("assets/thumbnail.jpg")
WIDTH, HEIGHT = 1200, 630


def main() -> None:
    corners = load_saved(OUT).get(PHOTO)
    models = load_models()
    if corners is None or models is None:
        print(f"need clicked corners for {PHOTO} in {OUT} and the models in models/")
        return
    result = read_board(cv2.imread(str(find_image(PHOTO))), models, build_board(), corners=corners)
    board = cv2.resize(result.rendered[:, :CANVAS], (HEIGHT, HEIGHT), interpolation=cv2.INTER_AREA)
    thumbnail = np.full((HEIGHT, WIDTH, 3), SEA_BGR, dtype=np.uint8)
    left = (WIDTH - HEIGHT) // 2
    thumbnail[:, left:left + HEIGHT] = board
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT), thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 90])
    pieces = sum(p.kind != "tile" for p in result.placements)
    print(f"saved {OUTPUT} ({WIDTH}x{HEIGHT}, {OUTPUT.stat().st_size // 1024} KB), {pieces} pieces drawn")


if __name__ == "__main__":
    main()
