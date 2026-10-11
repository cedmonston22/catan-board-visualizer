from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from board import Board, build_board, edge_midpoint
from click_corners import OUT, load_saved
from rectify import TILE_SIZE, compute_homography, find_image, warp

WEIGHTS = Path("runs/detect/catan_v2/weights/best.pt")
IMGSZ = 1024
CONF = 0.25
BASE_FRACTION = 0.75
MAX_SNAP = 0.5 * TILE_SIZE
OUTPUT = Path("outputs/snapped")


@dataclass
class Detection:
    piece: str
    conf: float
    box: tuple[float, float, float, float]


@dataclass
class Placement:
    piece: str
    conf: float
    kind: str
    target: int
    distance: float


def detect(model: YOLO, image_path: Path) -> list[Detection]:
    result = model.predict(str(image_path), imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    boxes = result.boxes.xyxy.tolist()
    classes = result.boxes.cls.tolist()
    confs = result.boxes.conf.tolist()
    return [
        Detection(piece=result.names[int(cls)], conf=conf, box=tuple(box))
        for box, cls, conf in zip(boxes, classes, confs)
    ]


def anchor_point(det: Detection) -> tuple[float, float]:
    x1, y1, x2, y2 = det.box
    cx = (x1 + x2) / 2
    if det.piece == "road":
        return cx, (y1 + y2) / 2
    return cx, y1 + BASE_FRACTION * (y2 - y1)


def to_board(points: list[tuple[float, float]], homography: np.ndarray) -> np.ndarray:
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    src = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(src, homography).reshape(-1, 2)


def nearest(point: np.ndarray, candidates: np.ndarray) -> tuple[int, float]:
    diff = candidates - point
    dist = np.linalg.norm(diff, axis=1)
    i = np.argmin(dist)
    return(int(i), float(dist[i]))


def snap(detections: list[Detection], homography: np.ndarray, board: Board) -> list[Placement]:
    detect_list = []
    for det in detections:
        detect_list.append(anchor_point(det))
    points = to_board(detect_list, homography)
    vertices = np.array(board.vertices)
    edge_mp_list = []
    for e in range(len(board.edges)):
        edge_mp_list.append(edge_midpoint(board,e))
    edge_mp = np.array(edge_mp_list)
    tiles = np.array(board.tiles)
    
    piece_kind = {"settlement": "vertex", "city": "vertex", "road": "edge", "robber": "tile"}
    candidates_for = {"vertex": vertices, "edge": edge_mp, "tile": tiles}
    
    placements = []
    for detect, point in zip(detections, points):
        kind = piece_kind[detect.piece]
        index, distance = nearest(point, candidates_for[kind])
        if distance <= MAX_SNAP:
            placements.append(Placement(detect.piece, detect.conf, kind, index, distance))
    return placements


def resolve_conflicts(placements: list[Placement]) -> list[Placement]:
    best = {} #key is (kind, target)
    for p in placements:
        if (p.kind, p.target) not in best:
            best[p.kind, p.target] = p
        elif p.conf > best[p.kind, p.target].conf:
            best[p.kind, p.target] = p
    return list(best.values())


PIECE_COLORS = {
    "settlement": (0, 200, 0),
    "city": (255, 0, 255),
    "road": (0, 255, 255),
    "robber": (0, 0, 0),
}


def draw_placements(warped: np.ndarray, board: Board, placements: list[Placement]) -> np.ndarray:
    canvas = warped.copy()
    for a, b in board.edges:
        pa = tuple(int(v) for v in board.vertices[a])
        pb = tuple(int(v) for v in board.vertices[b])
        cv2.line(canvas, pa, pb, (200, 200, 200), 1)

    for p in placements:
        color = PIECE_COLORS[p.piece]
        if p.kind == "edge":
            a, b = board.edges[p.target]
            pa = tuple(int(v) for v in board.vertices[a])
            pb = tuple(int(v) for v in board.vertices[b])
            cv2.line(canvas, pa, pb, color, 6)
        elif p.kind == "vertex":
            x, y = (int(v) for v in board.vertices[p.target])
            if p.piece == "city":
                cv2.rectangle(canvas, (x - 11, y - 11), (x + 11, y + 11), color, -1)
            else:
                cv2.circle(canvas, (x, y), 10, color, -1)
            cv2.circle(canvas, (x, y), 10, (255, 255, 255), 1)
        else:
            x, y = (int(v) for v in board.tiles[p.target])
            cv2.circle(canvas, (x, y), 16, color, -1)
            cv2.circle(canvas, (x, y), 16, (255, 255, 255), 2)
    return canvas


def main() -> None:
    if not WEIGHTS.exists():
        print(f"no weights at {WEIGHTS}, train a model first")
        return
    corners = load_saved(OUT)
    if not corners:
        print(f"no corners in {OUT}, run click_corners.py first")
        return

    model = YOLO(str(WEIGHTS))
    board = build_board()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for pid, clicked in corners.items():
        image_path = find_image(pid)
        homography = compute_homography(clicked)
        detections = detect(model, image_path)
        snapped = snap(detections, homography, board)
        placements = resolve_conflicts(snapped)

        warped = warp(cv2.imread(str(image_path)), homography)
        cv2.imwrite(str(OUTPUT / f"{pid}.jpg"), draw_placements(warped, board, placements))

        counts = {piece: sum(p.piece == piece for p in placements) for piece in PIECE_COLORS}
        off_board = len(detections) - len(snapped)
        conflicts = len(snapped) - len(placements)
        summary = "  ".join(f"{piece}={n}" for piece, n in counts.items())
        print(f"{pid:12} {summary}  off_board={off_board}  conflicts={conflicts}")
    print(f"saved to {OUTPUT}")


if __name__ == "__main__":
    main()
