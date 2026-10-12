import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from data_prep.click_corners import QuitRequested, click_image
from pipeline.board import Board, build_board
from pipeline.corners import find_corners, load_corner_model
from pipeline.hexes import Hex, fill_unknown_numbers, load_hex_models, read_hexes
from pipeline.rectify import warp
from pipeline.render import render_board
from pipeline.score import PlayerScore, score
from pipeline.snap import COLOR_WEIGHTS, WEIGHTS, Placement, read_image

OUTPUT = Path("outputs/boards")


@dataclass
class Models:
    pieces: YOLO
    colors: YOLO
    corners: YOLO
    tiles: YOLO
    tokens: YOLO


@dataclass
class BoardResult:
    problem: str | None
    corners: list[list[float]] | None = None
    corners_source: str | None = None
    placements: list[Placement] = field(default_factory=list)
    hexes: list[Hex] = field(default_factory=list)
    scores: list[PlayerScore] = field(default_factory=list)
    rendered: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)


def load_models() -> Models | None:
    for weights in (WEIGHTS, COLOR_WEIGHTS):
        if not weights.exists():
            print(f"no model at {weights}, train it first")
            return None
    corner_model = load_corner_model()
    hex_models = load_hex_models()
    if corner_model is None or hex_models is None:
        return None
    return Models(YOLO(str(WEIGHTS)), YOLO(str(COLOR_WEIGHTS)), corner_model, *hex_models)


def read_board(image: np.ndarray, models: Models, board: Board, corners: list[list[float]] | None = None,
               title: str = "") -> BoardResult:
    source = "manual"
    if corners is None:
        found = find_corners(models.corners, models.tiles, image)
        if found.points is None:
            return BoardResult(problem=found.problem)
        corners, source = found.points, "auto"

    reading = read_image(image, corners, models.pieces, models.colors, board)
    hexes = read_hexes(warp(image, reading.homography), models.tiles, models.tokens)
    for p in reading.placements:
        if p.kind == "tile":
            hexes[p.target].number = None
    unknown_before = sum(h.number is None and h.resource != "desert" for h in hexes)
    fill_unknown_numbers(hexes)
    unknown_after = sum(h.number is None and h.resource != "desert" for h in hexes)

    warnings = []
    if source == "manual":
        warnings.append("corners were placed by hand")
    if unknown_after:
        warnings.append(f"{unknown_after} number token(s) could not be read")
    deduced = unknown_before - unknown_after

    scores = score(board, reading.placements)
    rendered = render_board(board, hexes, reading.placements, scores, title)
    result = BoardResult(None, corners, source, reading.placements, hexes, scores, rendered, warnings)
    if deduced:
        result.warnings.append(f"{deduced} hidden number(s) deduced from the token counts")
    return result


def summary(result: BoardResult) -> str:
    counts = {piece: sum(p.piece == piece for p in result.placements) for piece in ("road", "settlement", "city")}
    robber = next((result.hexes[p.target].resource for p in result.placements if p.kind == "tile"), None)
    lines = [
        f"corners: {result.corners_source}",
        f"pieces:  {counts['road']} roads, {counts['settlement']} settlements, {counts['city']} cities"
        + (f", robber on {robber}" if robber else ""),
        "",
    ]
    for s in result.scores:
        badge = "  + Longest Road" if s.has_longest_road else ""
        lines.append(f"  {s.color:7} {s.points:2} pts   {s.settlements} settlements, {s.cities} cities, "
                     f"{s.roads} roads, longest {s.longest_road}{badge}")
    lines += [f"note: {w}" for w in result.warnings]
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m pipeline.read_board path/to/photo.jpg")
        return
    image_path = Path(sys.argv[1])
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"could not read an image at {image_path}")
        return
    models = load_models()
    if models is None:
        return
    board = build_board()

    result = read_board(image, models, board, title=image_path.stem)
    if result.problem:
        print(f"automatic corners failed ({result.problem}), click the 6 corners instead")
        try:
            clicked = click_image(image_path)
        except QuitRequested:
            clicked = None
        finally:
            cv2.destroyAllWindows()
        if clicked is None:
            print("no corners, stopping")
            return
        result = read_board(image, models, board, corners=clicked, title=image_path.stem)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / f"{image_path.stem}.jpg"
    cv2.imwrite(str(out_path), result.rendered)
    print(summary(result))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
