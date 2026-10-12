import math
from pathlib import Path

import cv2
import numpy as np

from pipeline.board import Board, build_board, hex_corners
from pipeline.hexes import Hex, load_hex_models, read_hexes, fill_unknown_numbers
from pipeline.rectify import CANVAS, CENTER, TILE_SIZE, warp
from pipeline.score import PlayerScore, score
from pipeline.snap import PLAYER_BGR, Placement, load_pipeline, read_photo

OUTPUT = Path("outputs/rendered")
SEA_BGR = (190, 130, 40)
FRAME_BGR = (120, 190, 225)
OUTLINE_BGR = (30, 30, 30)
TOKEN_BGR = (200, 235, 245)
RED_NUMBER_BGR = (30, 30, 200)
RESOURCE_BGR = {
    "wood": (40, 110, 40),
    "sheep": (90, 205, 130),
    "wheat": (40, 200, 240),
    "brick": (50, 85, 185),
    "ore": (150, 135, 125),
    "desert": (150, 205, 230),
}
PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
PANEL_WIDTH = 460


def board_outline(radius: float) -> np.ndarray:
    points = []
    for k in range(6):
        angle = math.radians(90 - 60 * k)
        points.append((CENTER[0] + radius * math.cos(angle), CENTER[1] - radius * math.sin(angle)))
    return np.array(points, dtype=np.int32)


def draw_tiles(canvas: np.ndarray, board: Board, hexes: list[Hex]) -> None:
    for center, h in zip(board.tiles, hexes):
        corners = np.array(hex_corners(center), dtype=np.int32)
        cv2.fillPoly(canvas, [corners], RESOURCE_BGR[h.resource], cv2.LINE_AA)
        cv2.polylines(canvas, [corners], True, OUTLINE_BGR, 2, cv2.LINE_AA)
        cv2.putText(canvas, h.resource, (int(center[0]) - 26, int(center[1]) + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, OUTLINE_BGR, 1, cv2.LINE_AA)
        if h.resource == "desert":
            continue
        x, y = int(center[0]), int(center[1])
        cv2.circle(canvas, (x, y), 30, TOKEN_BGR, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 30, OUTLINE_BGR, 1, cv2.LINE_AA)
        color = RED_NUMBER_BGR if h.number in (6, 8) else OUTLINE_BGR
        text = "?" if h.number is None else str(h.number)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        cv2.putText(canvas, text, (x - tw // 2, y + 8), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2, cv2.LINE_AA)
        if h.number is None:
            continue
        pips = PIPS[h.number]
        for i in range(pips):
            cv2.circle(canvas, (int(x + (i - (pips - 1) / 2) * 7), y + 18), 2, color, -1, cv2.LINE_AA)


def house(x: int, y: int, size: int) -> np.ndarray:
    return np.array([(x - size, y + size), (x - size, y - size // 3), (x, y - size),
                     (x + size, y - size // 3), (x + size, y + size)], dtype=np.int32)


def city(x: int, y: int, size: int) -> np.ndarray:
    return np.array([(x - size * 1.4, y + size), (x - size * 1.4, y - size * 0.2), (x - size * 0.7, y - size * 0.9),
                     (x, y - size * 0.2), (x, y - size * 1.3), (x + size * 1.2, y - size * 1.3),
                     (x + size * 1.2, y + size)], dtype=np.int32)


def draw_pieces(canvas: np.ndarray, board: Board, placements: list[Placement]) -> None:
    for p in placements:
        if p.kind != "edge":
            continue
        a, b = board.edges[p.target]
        pa = tuple(int(v) for v in board.vertices[a])
        pb = tuple(int(v) for v in board.vertices[b])
        cv2.line(canvas, pa, pb, OUTLINE_BGR, 14, cv2.LINE_AA)
        cv2.line(canvas, pa, pb, PLAYER_BGR.get(p.color, (128, 128, 128)), 9, cv2.LINE_AA)

    for p in placements:
        if p.kind == "vertex":
            x, y = (int(v) for v in board.vertices[p.target])
            shape = city(x, y, 13) if p.piece == "city" else house(x, y, 12)
            cv2.fillPoly(canvas, [shape], PLAYER_BGR.get(p.color, (128, 128, 128)), cv2.LINE_AA)
            cv2.polylines(canvas, [shape], True, OUTLINE_BGR, 2, cv2.LINE_AA)
        elif p.kind == "tile":
            x, y = (int(v) for v in board.tiles[p.target])
            cv2.ellipse(canvas, (x - 34, y + 4), (11, 20), 0, 0, 360, (60, 60, 60), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x - 34, y - 18), 9, (60, 60, 60), -1, cv2.LINE_AA)


def draw_scores(height: int, scores: list[PlayerScore], title: str) -> np.ndarray:
    panel = np.full((height, PANEL_WIDTH, 3), 250, dtype=np.uint8)
    cv2.putText(panel, title, (24, 56), cv2.FONT_HERSHEY_DUPLEX, 0.9, OUTLINE_BGR, 1, cv2.LINE_AA)
    y = 120
    for s in scores:
        cv2.rectangle(panel, (24, y - 28), (60, y + 8), PLAYER_BGR[s.color], -1)
        cv2.rectangle(panel, (24, y - 28), (60, y + 8), OUTLINE_BGR, 2)
        cv2.putText(panel, f"{s.color}  {s.points} pts", (76, y), cv2.FONT_HERSHEY_DUPLEX, 0.85, OUTLINE_BGR, 1,
                    cv2.LINE_AA)
        lines = [f"{s.settlements} settlements, {s.cities} cities", f"{s.roads} roads, longest {s.longest_road}"]
        if s.has_longest_road:
            lines.append("Longest Road +2")
        for j, line in enumerate(lines):
            cv2.putText(panel, line, (76, y + 30 + 26 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1,
                        cv2.LINE_AA)
        y += 70 + 26 * len(lines)
    return panel


def render_board(board: Board, hexes: list[Hex], placements: list[Placement], scores: list[PlayerScore],
                 title: str) -> np.ndarray:
    canvas = np.full((CANVAS, CANVAS, 3), SEA_BGR, dtype=np.uint8)
    cv2.fillPoly(canvas, [board_outline(2.5 * math.sqrt(3) * TILE_SIZE + 30)], FRAME_BGR, cv2.LINE_AA)
    draw_tiles(canvas, board, hexes)
    draw_pieces(canvas, board, placements)
    return np.hstack([canvas, draw_scores(CANVAS, scores, title)])


def main() -> None:
    loaded = load_pipeline()
    hex_models = load_hex_models()
    if loaded is None or hex_models is None:
        return
    model, color_model, corners = loaded
    tile_model, token_model = hex_models
    board = build_board()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for pid, clicked in corners.items():
        reading = read_photo(pid, clicked, model, color_model, board)
        hexes = read_hexes(warp(reading.image, reading.homography), tile_model, token_model)
        for p in reading.placements:
            if p.kind == "tile":
                hexes[p.target].number = None
        for p in reading.placements:
            if p.kind == "tile":
                hexes[p.target].number = None
        fill_unknown_numbers(hexes)        
        scores = score(board, reading.placements)
        image = render_board(board, hexes, reading.placements, scores, pid)
        cv2.imwrite(str(OUTPUT / f"{pid}.jpg"), image)
        summary = " ".join(f"{h.resource[:2]}{h.number or ''}" for h in hexes)
        print(f"{pid:12} {summary}")
    print(f"saved to {OUTPUT}")


if __name__ == "__main__":
    main()
