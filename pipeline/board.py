import math
from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.rectify import OUTPUT, TILE_SIZE, tile_centers


@dataclass
class Board:
    tiles: list[tuple[float, float]]
    vertices: list[tuple[float, float]]
    edges: list[tuple[int, int]]
    tile_vertices: list[list[int]]


def hex_corners(center: tuple[float, float]) -> list[tuple[float, float]]:
    corners = []
    for k in range(6):
        angle = math.radians(60 * k)
        corners.append((center[0] + TILE_SIZE * math.cos(angle), center[1] + TILE_SIZE * math.sin(angle)))
    return corners


def vertex_key(point: tuple[float, float]) -> tuple[float, float]:
    return((round(point[0], 1), round(point[1],1)))

def build_board() -> Board:
    vertices = []
    key_to_id = {}
    tile_vertices = []
    edges = set()
    tiles = tile_centers()
    
    for tile in tiles:
        ids = []
        curr_corners = hex_corners(tile)
        for corner in curr_corners:
            key = vertex_key(corner)
            if key not in key_to_id:
                key_to_id[key] = len(vertices)
                vertices.append(corner)
            ids.append(key_to_id[key])
        tile_vertices.append(ids)
        for i in range(6):
            a = ids[i]
            b = ids[(i + 1) % 6]
            edges.add((min(a,b), max(a,b)))
        
    return Board(tiles, vertices, sorted(edges), tile_vertices)
        
        


def edge_midpoint(board: Board, edge_id: int) -> tuple[float, float]:
    a, b = board.edges[edge_id]
    (ax, ay), (bx, by) = board.vertices[a], board.vertices[b]
    return ((ax + bx) / 2, (ay + by) / 2)


def draw_board(image: np.ndarray, board: Board) -> np.ndarray:
    canvas = image.copy()
    for edge_id, (a, b) in enumerate(board.edges):
        pa = tuple(int(v) for v in board.vertices[a])
        pb = tuple(int(v) for v in board.vertices[b])
        cv2.line(canvas, pa, pb, (0, 255, 255), 1)
        mx, my = edge_midpoint(board, edge_id)
        cv2.putText(canvas, str(edge_id), (int(mx) - 8, int(my) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    for vertex_id, (x, y) in enumerate(board.vertices):
        cv2.circle(canvas, (int(x), int(y)), 4, (0, 0, 255), -1)
        cv2.putText(canvas, str(vertex_id), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    for tile_id, (x, y) in enumerate(board.tiles):
        cv2.putText(canvas, f"T{tile_id}", (int(x) - 12, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return canvas


def main() -> None:
    board = build_board()
    print(f"{len(board.tiles)} tiles, {len(board.vertices)} vertices, {len(board.edges)} edges")

    rectified = sorted(OUTPUT.glob("*.jpg"))
    if not rectified:
        print(f"no images in {OUTPUT}, run rectify.py first")
        return
    out_path = OUTPUT.parent / "board_layout.jpg"
    cv2.imwrite(str(out_path), draw_board(cv2.imread(str(rectified[0])), board))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
