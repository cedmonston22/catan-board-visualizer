import base64
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from pipeline.board import build_board, edge_midpoint
from pipeline.corners import clockwise_from_top, is_convex_hexagon
from pipeline.read_board import BoardResult, Models, load_models, read_board
from pipeline.rectify import CANVAS, TILE_SIZE
from pipeline.render import SEA_BGR, render_board
from pipeline.score import BUILDING_POINTS, LONGEST_ROAD_POINTS, PLAYERS, score
from pipeline.snap import PLAYER_BGR, Placement

logger = logging.getLogger("catan_app")
logging.basicConfig(level=logging.INFO)

GUIDE = "assets/corner_guide.jpg"
EXAMPLES = sorted(Path("assets/examples").glob("*.jpg"))
DISPLAY_MAX = 1280
NUM_CORNERS = 6
SLIDE_SECONDS = 0.9
SELECT_RADIUS = 0.4 * TILE_SIZE
BUILDINGS = ["settlement", "city"]
HIGHLIGHT_BGR = (0, 230, 255)

CSS = f"""
#title h1 {{ font-size: 1.4rem; margin: 0; }}
#taps img, #editor img {{ max-height: 62vh; width: 100%; object-fit: contain; }}
#actions, #edit-actions {{ flex-wrap: nowrap; gap: 8px; }}
#guide img {{ width: 100%; object-fit: contain; }}
#stage .html-container {{ padding: 0 !important; }}
#stage .stage {{ position: relative; overflow: hidden; border-radius: 8px; background: #111; }}
#stage .stage .photo {{ display: block; width: 100%; max-height: 62vh; object-fit: contain; }}
#stage .stage .board-layer {{
  position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  background: rgb({SEA_BGR[2]}, {SEA_BGR[1]}, {SEA_BGR[0]});
  animation: slide-in {SLIDE_SECONDS}s ease-out both;
}}
#stage .stage .board-layer img {{ max-width: 100%; max-height: 100%; }}
@keyframes slide-in {{ from {{ transform: translateX(-100%); }} to {{ transform: translateX(0); }} }}
#info .scores {{ display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 1.05rem; font-weight: 600; }}
#info .dot {{ display: inline-block; width: 0.8em; height: 0.8em; border-radius: 50%; border: 1px solid #555; margin-right: 4px; }}
#info .notes {{ font-size: 0.8rem; opacity: 0.75; margin-top: 4px; }}
#info .breakdown {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; table-layout: fixed; border: none !important; }}
#info .breakdown th, #info .breakdown td {{ padding: 5px 2px !important; text-align: center; vertical-align: middle;
  border: none !important; border-bottom: 1px solid rgba(128, 128, 128, 0.3) !important; background: none !important; }}
#info .breakdown thead th {{ font-size: 0.68rem; font-weight: 500; opacity: 0.7; line-height: 1.2; }}
#info .breakdown thead th:first-child, #info .breakdown tbody th {{ width: 30%; }}
#info .breakdown tbody th {{ text-align: left; white-space: nowrap; overflow: hidden; }}
#info .breakdown .n {{ display: block; font-weight: 600; }}
#info .breakdown .sub {{ display: block; font-size: 0.68rem; opacity: 0.7; }}
#info .breakdown .won .n {{ color: #f5b400; }}
#info .breakdown .total {{ font-size: 1.25rem; font-weight: 700; }}
"""

VIEW_NAMES = ["photo", "taps", "editor", "stage", "info", "state", "undo", "read_corners", "fix", "edit", "new",
              "guide", "piece", "color", "save", "remove", "done", "examples"]


@dataclass
class Session:
    image_bgr: np.ndarray
    display_bgr: np.ndarray
    scale: float
    taps: list[tuple[int, int]] = field(default_factory=list)
    result: BoardResult | None = None
    seconds: float = 0.0
    selected: tuple[str, int] | None = None
    last_color: str | None = None
    edits: int = 0


def load_app_models() -> Models:
    models = load_models()
    if models is None:
        raise RuntimeError("models are missing from models/, promote them with python -m pipeline.promote")
    for model in (models.pieces, models.colors, models.corners, models.tiles, models.tokens):
        model.overrides["device"] = "cpu"
    return models


MODELS = load_app_models()
BOARD = build_board()
VERTEX_POINTS = np.array(BOARD.vertices, dtype=np.float32)
EDGE_POINTS = np.array([edge_midpoint(BOARD, i) for i in range(len(BOARD.edges))], dtype=np.float32)


def analyze(image_bgr: np.ndarray, corners: list[list[float]] | None = None) -> BoardResult:
    return read_board(image_bgr, MODELS, BOARD, corners=corners)


def make_session(image_rgb: np.ndarray) -> Session:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    scale = min(1.0, DISPLAY_MAX / max(image_bgr.shape[:2]))
    display = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image_bgr
    return Session(image_bgr, display, scale)


def run(session: Session, corners: list[list[float]] | None) -> BoardResult:
    start = time.perf_counter()
    try:
        result = analyze(session.image_bgr, corners)
    except Exception as error:
        logger.exception("reading the board failed")
        raise gr.Error(f"Couldn't read this photo: {error}") from error
    session.seconds = time.perf_counter() - start
    h, w = session.image_bgr.shape[:2]
    logger.info("read %dx%d photo in %.2f s, corners=%s, problem=%s", w, h, session.seconds,
                "manual" if corners else "auto", result.problem)
    if result.problem is None:
        session.result, session.selected, session.edits = result, None, 0
    return result


def data_url(image_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise gr.Error("Couldn't encode the image for display.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def board_only(result: BoardResult) -> np.ndarray:
    return result.rendered[:, :CANVAS]


def stage_html(session: Session) -> str:
    return (f'<div class="stage"><img class="photo" src="{data_url(session.display_bgr)}" alt="board photo">'
            f'<div class="board-layer"><img src="{data_url(board_only(session.result))}" alt="regenerated board">'
            f'</div></div>')


def css_color(color: str) -> str:
    b, g, r = PLAYER_BGR.get(color, (128, 128, 128))
    return f"rgb({r}, {g}, {b})"


def notes_html(session: Session) -> str:
    edited = [f"{session.edits} edit(s) by hand"] if session.edits else []
    notes = " · ".join(session.result.warnings + edited + [f"read in {session.seconds:.1f} s"])
    return f'<div class="notes">{notes}</div>'


def scores_html(session: Session) -> str:
    players = "".join(
        f'<span><span class="dot" style="background:{css_color(s.color)}"></span>{s.color} {s.points}'
        f'{" (Longest Road)" if s.has_longest_road else ""}</span>'
        for s in session.result.scores
    ) or "<span>No pieces found on the board</span>"
    return f'<div class="scores">{players}</div>{notes_html(session)}'


def cell(count: int, points: int, unit: str) -> str:
    earned = f"+{points}" if points else "0"
    return f'<td><span class="n">{count}</span><span class="sub">{earned} {unit}</span></td>'


def breakdown_html(session: Session) -> str:
    scores = session.result.scores
    if not scores:
        return f'<div class="scores">No pieces found on the board</div>{notes_html(session)}'
    rows = []
    for s in scores:
        bonus = LONGEST_ROAD_POINTS if s.has_longest_road else 0
        road = (f'<td class="{"won" if bonus else ""}"><span class="n">{s.longest_road}{" ★" if bonus else ""}</span>'
                f'<span class="sub">{f"+{bonus}" if bonus else "0"} pts</span></td>')
        rows.append(f'<tr><th><span class="n"><span class="dot" style="background:{css_color(s.color)}"></span>'
                    f'{s.color}</span><span class="sub">{s.roads} road pieces</span></th>'
                    f'{cell(s.settlements, s.settlements * BUILDING_POINTS["settlement"], "pts")}'
                    f'{cell(s.cities, s.cities * BUILDING_POINTS["city"], "pts")}{road}'
                    f'<td class="total">{s.points}</td></tr>')
    header = "<tr><th></th><th>Settle&shy;ments</th><th>Cities</th><th>Longest road</th><th>Total</th></tr>"
    return f'<table class="breakdown"><thead>{header}</thead><tbody>{"".join(rows)}</tbody></table>{notes_html(session)}'


def tap_info(session: Session, message: str = "") -> str:
    count = len(session.taps)
    prompt = (f"Tap the {NUM_CORNERS} points where the sandy beach meets the sea, like the yellow dots below "
              f"({count}/{NUM_CORNERS})"
              if count < NUM_CORNERS
              else "All 6 corners placed. Read the board, or undo to move one.")
    extra = f'<div class="notes">{message}</div>' if message else ""
    return f"<div><b>{prompt}</b></div>{extra}"


def tap_image(session: Session) -> np.ndarray:
    canvas = session.display_bgr.copy()
    radius = max(6, round(max(canvas.shape[:2]) / 70))
    if len(session.taps) == NUM_CORNERS:
        outline = clockwise_from_top(np.array(session.taps, dtype=np.float32)).astype(np.int32)
        cv2.polylines(canvas, [outline], True, (0, 255, 255), max(2, radius // 3), cv2.LINE_AA)
    for i, (x, y) in enumerate(session.taps, start=1):
        cv2.circle(canvas, (x, y), radius, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), radius, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, str(i), (x + radius, y - radius), cv2.FONT_HERSHEY_SIMPLEX, radius / 12, (0, 0, 0), 4,
                    cv2.LINE_AA)
        cv2.putText(canvas, str(i), (x + radius, y - radius), cv2.FONT_HERSHEY_SIMPLEX, radius / 12, (255, 255, 255),
                    2, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def nearest_spot(x: float, y: float) -> tuple[str, int] | None:
    point = np.array([x, y], dtype=np.float32)
    vertex_distances = np.linalg.norm(VERTEX_POINTS - point, axis=1)
    edge_distances = np.linalg.norm(EDGE_POINTS - point, axis=1)
    vertex, edge = int(vertex_distances.argmin()), int(edge_distances.argmin())
    if min(vertex_distances[vertex], edge_distances[edge]) > SELECT_RADIUS:
        return None
    return ("vertex", vertex) if vertex_distances[vertex] <= edge_distances[edge] else ("edge", edge)


def placement_at(session: Session, spot: tuple[str, int]) -> Placement | None:
    kind, target = spot
    return next((p for p in session.result.placements if p.kind == kind and p.target == target), None)


def spot_point(spot: tuple[str, int]) -> tuple[int, int]:
    kind, target = spot
    x, y = BOARD.vertices[target] if kind == "vertex" else edge_midpoint(BOARD, target)
    return int(x), int(y)


def edit_image(session: Session) -> np.ndarray:
    canvas = board_only(session.result).copy()
    if session.selected:
        center = spot_point(session.selected)
        cv2.circle(canvas, center, 26, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.circle(canvas, center, 26, HIGHLIGHT_BGR, 4, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def rescore(session: Session) -> None:
    result = session.result
    result.scores = score(BOARD, result.placements)
    result.rendered = render_board(BOARD, result.hexes, result.placements, result.scores, "")


def view(**shown: object) -> tuple:
    values = {name: gr.update(visible=False) for name in VIEW_NAMES}
    values["info"], values["state"] = gr.update(), gr.update()
    values.update(shown)
    return tuple(values[name] for name in VIEW_NAMES)


def upload_view() -> tuple:
    return view(photo=gr.update(value=None, visible=True), taps=gr.update(value=None, visible=False),
                editor=gr.update(value=None, visible=False), stage=gr.update(value="", visible=False),
                info="Take a photo of the whole board from above.", state=None,
                examples=gr.update(visible=True))


def tap_view(session: Session, message: str = "") -> tuple:
    return view(taps=gr.update(value=tap_image(session), visible=True), info=tap_info(session, message), state=session,
                undo=gr.update(visible=True, interactive=bool(session.taps)),
                read_corners=gr.update(visible=True, interactive=len(session.taps) == NUM_CORNERS),
                new=gr.update(visible=True), guide=gr.update(visible=True))


def result_view(session: Session) -> tuple:
    return view(stage=gr.update(value=stage_html(session), visible=True), info=breakdown_html(session), state=session,
                fix=gr.update(visible=True), edit=gr.update(visible=True), new=gr.update(visible=True))


def edit_view(session: Session, message: str = "") -> tuple:
    selected = session.selected
    if selected is None:
        prompt = message or "Tap a road spot or a building spot to add, change or remove a piece."
        return view(editor=gr.update(value=edit_image(session), visible=True),
                    info=f"{scores_html(session)}<div><b>{prompt}</b></div>", state=session,
                    done=gr.update(visible=True))
    current = placement_at(session, selected)
    is_vertex = selected[0] == "vertex"
    what = "building spot" if is_vertex else "road spot"
    status = f"{current.color or 'uncolored'} {current.piece}" if current else "empty"
    prompt = message or f"Selected {what} ({status}). Choose {'type and ' if is_vertex else ''}color, then save."
    piece_value = current.piece if current and current.piece in BUILDINGS else "settlement"
    color_value = current.color if current and current.color in PLAYERS else session.last_color
    return view(editor=gr.update(value=edit_image(session), visible=True),
                info=f"{scores_html(session)}<div><b>{prompt}</b></div>", state=session,
                piece=gr.update(visible=is_vertex, value=piece_value),
                color=gr.update(visible=True, value=color_value),
                save=gr.update(visible=True), remove=gr.update(visible=current is not None),
                done=gr.update(visible=True))


def on_upload_photo(image_rgb: np.ndarray | None) -> tuple:
    if image_rgb is None:
        return upload_view()
    session = make_session(image_rgb)
    result = run(session, None)
    if result.problem:
        return tap_view(session, f"Couldn't find the corners automatically ({result.problem}).")
    return result_view(session)


def on_select_tap(session: Session | None, evt: gr.SelectData) -> tuple:
    if session is None:
        return upload_view()
    if len(session.taps) < NUM_CORNERS and evt.index is not None:
        x, y = evt.index
        session.taps.append((int(x), int(y)))
    return tap_view(session)


def on_click_undo(session: Session | None) -> tuple:
    if session is None:
        return upload_view()
    if session.taps:
        session.taps.pop()
    return tap_view(session)


def on_click_read_corners(session: Session | None) -> tuple:
    if session is None:
        return upload_view()
    if len(session.taps) != NUM_CORNERS:
        return tap_view(session, f"Place all {NUM_CORNERS} corners first.")
    ordered = clockwise_from_top(np.array(session.taps, dtype=np.float32) / session.scale)
    if not is_convex_hexagon(ordered):
        return tap_view(session, "Those taps don't make a hexagon. Undo and tap the beach points shown in the example.")
    result = run(session, ordered.tolist())
    if result.problem:
        return tap_view(session, f"Couldn't read the board with these corners ({result.problem}).")
    return result_view(session)


def on_click_fix_corners(session: Session | None) -> tuple:
    if session is None:
        return upload_view()
    session.taps = []
    return tap_view(session)


def on_click_edit(session: Session | None) -> tuple:
    if session is None or session.result is None:
        return upload_view()
    session.selected = None
    return edit_view(session)


def on_select_spot(session: Session | None, evt: gr.SelectData) -> tuple:
    if session is None or session.result is None:
        return upload_view()
    spot = nearest_spot(*evt.index) if evt.index is not None else None
    if spot is None:
        session.selected = None
        return edit_view(session, "Tap closer to a corner (building) or the middle of an edge (road).")
    session.selected = spot
    return edit_view(session)


def on_click_save(session: Session | None, piece: str | None, color: str | None) -> tuple:
    if session is None or session.result is None:
        return upload_view()
    if session.selected is None:
        return edit_view(session)
    if color not in PLAYERS:
        return edit_view(session, "Pick a color first.")
    kind, target = session.selected
    if kind == "vertex" and piece not in BUILDINGS:
        return edit_view(session, "Pick settlement or city first.")
    new_piece = piece if kind == "vertex" else "road"
    current = placement_at(session, session.selected)
    placements = [p for p in session.result.placements if p is not current]
    placements.append(Placement(new_piece, 1.0, kind, target, 0.0, color))
    session.result.placements = placements
    session.last_color, session.edits, session.selected = color, session.edits + 1, None
    rescore(session)
    return edit_view(session, f"Saved {color} {new_piece}. Tap another spot, or Done.")


def on_click_remove(session: Session | None) -> tuple:
    if session is None or session.result is None:
        return upload_view()
    current = placement_at(session, session.selected) if session.selected else None
    if current is None:
        return edit_view(session)
    session.result.placements = [p for p in session.result.placements if p is not current]
    session.edits, session.selected = session.edits + 1, None
    rescore(session)
    return edit_view(session, f"Removed {current.color or 'uncolored'} {current.piece}. Tap another spot, or Done.")


def on_click_done(session: Session | None) -> tuple:
    if session is None or session.result is None:
        return upload_view()
    session.selected = None
    return result_view(session)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Catan Board Reader") as demo:
        gr.Markdown("# Catan Board Reader", elem_id="title")
        state = gr.State(None)
        photo = gr.Image(type="numpy", sources=["upload"], image_mode="RGB", show_label=False, buttons=[],
                         elem_id="photo")
        taps = gr.Image(type="numpy", interactive=False, show_label=False, buttons=[], visible=False, elem_id="taps")
        editor = gr.Image(type="numpy", interactive=False, show_label=False, buttons=[], visible=False,
                          elem_id="editor")
        stage = gr.HTML(elem_id="stage", visible=False, padding=False)
        info = gr.HTML("Take a photo of the whole board from above.", elem_id="info")
        examples_box = gr.Column(visible=bool(EXAMPLES), elem_id="examples")
        piece = gr.Radio(BUILDINGS, label="Type", visible=False)
        color = gr.Radio(PLAYERS, label="Color", visible=False)
        with gr.Row(elem_id="edit-actions"):
            save_button = gr.Button("Save", variant="primary", size="sm", visible=False, min_width=0)
            remove_button = gr.Button("Remove", size="sm", visible=False, min_width=0)
            done_button = gr.Button("Done", size="sm", visible=False, min_width=0)
        with gr.Row(elem_id="actions"):
            undo_button = gr.Button("Undo", size="sm", visible=False, min_width=0)
            corners_button = gr.Button("Read board", variant="primary", size="sm", visible=False, min_width=0)
            fix_button = gr.Button("Fix corners", size="sm", visible=False, min_width=0)
            edit_button = gr.Button("Edit pieces", size="sm", visible=False, min_width=0)
            new_button = gr.Button("New photo", size="sm", visible=False, min_width=0)
        guide = gr.Image(GUIDE, interactive=False, show_label=False, buttons=[], visible=False, elem_id="guide")

        components = {"photo": photo, "taps": taps, "editor": editor, "stage": stage, "info": info, "state": state,
                      "undo": undo_button, "read_corners": corners_button, "fix": fix_button, "edit": edit_button,
                      "new": new_button, "guide": guide, "piece": piece, "color": color, "save": save_button,
                      "remove": remove_button, "done": done_button, "examples": examples_box}
        outputs = [components[name] for name in VIEW_NAMES]
        if EXAMPLES:
            with examples_box:
                gr.Examples([[str(path)] for path in EXAMPLES], inputs=photo, outputs=outputs, fn=on_upload_photo,
                            run_on_click=True, cache_examples=False, label="No board handy? Tap an example")
        photo.upload(on_upload_photo, inputs=photo, outputs=outputs, api_name="read")
        photo.clear(upload_view, outputs=outputs, api_name=False)
        taps.select(on_select_tap, inputs=state, outputs=outputs, api_name=False)
        undo_button.click(on_click_undo, inputs=state, outputs=outputs, api_name=False)
        corners_button.click(on_click_read_corners, inputs=state, outputs=outputs, api_name="read_corners")
        fix_button.click(on_click_fix_corners, inputs=state, outputs=outputs, api_name=False)
        edit_button.click(on_click_edit, inputs=state, outputs=outputs, api_name=False)
        editor.select(on_select_spot, inputs=state, outputs=outputs, api_name=False)
        save_button.click(on_click_save, inputs=[state, piece, color], outputs=outputs, api_name=False)
        remove_button.click(on_click_remove, inputs=state, outputs=outputs, api_name=False)
        done_button.click(on_click_done, inputs=state, outputs=outputs, api_name=False)
        new_button.click(upload_view, outputs=outputs, api_name=False)
    return demo


if __name__ == "__main__":
    build_ui().launch(share="--share" in sys.argv, css=CSS)
