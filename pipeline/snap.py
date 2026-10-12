from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from pipeline.board import Board, build_board, edge_midpoint
from data_prep.click_corners import OUT, load_saved
from data_prep.extract_crops import PAD, crop_box
from pipeline.rectify import TILE_SIZE, compute_homography, find_image, warp
from pipeline.weights import weights_path

WEIGHTS = weights_path("pieces")
COLOR_WEIGHTS = weights_path("colors")
COLOR_IMGSZ = 128
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
    color: str | None = None


@dataclass
class Placement:
    piece: str
    conf: float
    kind: str
    target: int
    distance: float
    color: str | None = None


def detect(model: YOLO, image: np.ndarray) -> list[Detection]:
    result = model.predict(image, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    boxes = result.boxes.xyxy.tolist()
    classes = result.boxes.cls.tolist()
    confs = result.boxes.conf.tolist()
    return [
        Detection(piece=result.names[int(cls)], conf=conf, box=tuple(box))
        for box, cls, conf in zip(boxes, classes, confs)
    ]


def color_detections(detections: list[Detection], image: np.ndarray, color_model: YOLO) -> None:
    targets, crops = [], []
    for det in detections:
        if det.piece == "robber":
            continue
        crop = crop_box(image, tuple(int(v) for v in det.box), PAD)
        if crop.size == 0:
            continue
        targets.append(det)
        crops.append(crop)
    if not crops:
        return
    results = color_model.predict(crops, imgsz=COLOR_IMGSZ, verbose=False)
    for det, result in zip(targets, results):
        det.color = result.names[result.probs.top1]


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
            placements.append(Placement(detect.piece, detect.conf, kind, index, distance, color=detect.color))
    return placements


def resolve_conflicts(placements: list[Placement]) -> list[Placement]:
    best = {} #key is (kind, target)
    for p in placements:
        if (p.kind, p.target) not in best:
            best[p.kind, p.target] = p
        elif p.conf > best[p.kind, p.target].conf:
            best[p.kind, p.target] = p
    return list(best.values())


PIECES = ["settlement", "city", "road", "robber"]
PLAYER_BGR = {
    "red": (40, 40, 220),
    "blue": (200, 90, 30),
    "white": (245, 245, 245),
    "orange": (20, 140, 255),
}
UNKNOWN_BGR = (128, 128, 128)
OUTLINE_BGR = (20, 20, 20)


def player_summary(placements: list[Placement]) -> dict[str, dict[str, int]]:
    res = {}
    player_pieces = {"white" : [], "blue": [], "red": [], "orange": []}
    for place in placements:
        if place.color is not None:
            player_pieces[place.color].append(place)
    
    for color, pieces in player_pieces.items():
        piece_type_count = {"settlement": 0, "city": 0, "road": 0}
        if not pieces:
            continue
        for p in pieces:
            piece_type_count[p.piece] += 1
        res[color] = piece_type_count
    return res
    


def draw_placements(warped: np.ndarray, board: Board, placements: list[Placement]) -> np.ndarray:
    canvas = warped.copy()
    for a, b in board.edges:
        pa = tuple(int(v) for v in board.vertices[a])
        pb = tuple(int(v) for v in board.vertices[b])
        cv2.line(canvas, pa, pb, (200, 200, 200), 1)

    for p in sorted(placements, key=lambda p: p.kind != "edge"):
        color = PLAYER_BGR.get(p.color, UNKNOWN_BGR)
        if p.kind == "edge":
            a, b = board.edges[p.target]
            pa = tuple(int(v) for v in board.vertices[a])
            pb = tuple(int(v) for v in board.vertices[b])
            cv2.line(canvas, pa, pb, OUTLINE_BGR, 10)
            cv2.line(canvas, pa, pb, color, 6)
        elif p.kind == "vertex":
            x, y = (int(v) for v in board.vertices[p.target])
            if p.piece == "city":
                cv2.rectangle(canvas, (x - 12, y - 12), (x + 12, y + 12), color, -1)
                cv2.rectangle(canvas, (x - 12, y - 12), (x + 12, y + 12), OUTLINE_BGR, 2)
            else:
                cv2.circle(canvas, (x, y), 10, color, -1)
                cv2.circle(canvas, (x, y), 10, OUTLINE_BGR, 2)
        else:
            x, y = (int(v) for v in board.tiles[p.target])
            cv2.circle(canvas, (x, y), 16, (0, 0, 0), -1)
            cv2.circle(canvas, (x, y), 16, (255, 255, 255), 2)
    return canvas


@dataclass
class Reading:
    image: np.ndarray
    homography: np.ndarray
    detections: list[Detection]
    snapped: list[Placement]
    placements: list[Placement]


def load_pipeline() -> tuple[YOLO, YOLO, dict[str, list[list[float]]]] | None:
    if not WEIGHTS.exists():
        print(f"no weights at {WEIGHTS}, promote one with python -m pipeline.promote pieces path/to/best.pt")
        return None
    if not COLOR_WEIGHTS.exists():
        print(f"no color classifier at {COLOR_WEIGHTS}, promote one with python -m pipeline.promote colors path/to/best.pt")
        return None
    corners = load_saved(OUT)
    if not corners:
        print(f"no corners in {OUT}, run click_corners.py first")
        return None
    return YOLO(str(WEIGHTS)), YOLO(str(COLOR_WEIGHTS)), corners


def read_image(image: np.ndarray, corners: list[list[float]], model: YOLO, color_model: YOLO, board: Board) -> Reading:
    homography = compute_homography(corners)
    detections = detect(model, image)
    color_detections(detections, image, color_model)
    snapped = snap(detections, homography, board)
    return Reading(image, homography, detections, snapped, resolve_conflicts(snapped))


def read_photo(pid: str, clicked: list[list[float]], model: YOLO, color_model: YOLO, board: Board) -> Reading:
    return read_image(cv2.imread(str(find_image(pid))), clicked, model, color_model, board)


def main() -> None:
    loaded = load_pipeline()
    if loaded is None:
        return
    model, color_model, corners = loaded
    board = build_board()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for pid, clicked in corners.items():
        reading = read_photo(pid, clicked, model, color_model, board)
        placements = reading.placements

        warped = warp(reading.image, reading.homography)
        cv2.imwrite(str(OUTPUT / f"{pid}.jpg"), draw_placements(warped, board, placements))

        counts = {piece: sum(p.piece == piece for p in placements) for piece in PIECES}
        off_board = len(reading.detections) - len(reading.snapped)
        conflicts = len(reading.snapped) - len(placements)
        summary = "  ".join(f"{piece}={n}" for piece, n in counts.items())
        print(f"{pid:12} {summary}  off_board={off_board}  conflicts={conflicts}")
        for color, pieces in player_summary(placements).items():
            print(f"    {color:7} {pieces['settlement']} settlements, {pieces['city']} cities, {pieces['road']} roads")
    print(f"saved to {OUTPUT}")


if __name__ == "__main__":
    main()
