import base64
import logging
import sys
import time
from dataclasses import dataclass, field

import cv2
import gradio as gr
import numpy as np
import spaces
import torch

from pipeline.board import build_board
from pipeline.corners import clockwise_from_top, is_convex_hexagon
from pipeline.read_board import BoardResult, Models, load_models, read_board
from pipeline.rectify import CANVAS
from pipeline.render import SEA_BGR
from pipeline.snap import PLAYER_BGR

logger = logging.getLogger("catan_app")
logging.basicConfig(level=logging.INFO)

GUIDE = "assets/corner_guide.jpg"
DISPLAY_MAX = 1280
NUM_CORNERS = 6
SLIDE_SECONDS = 0.9

CSS = f"""
#title h1 {{ font-size: 1.4rem; margin: 0; }}
#taps img {{ max-height: 62vh; width: 100%; object-fit: contain; }}
#actions {{ flex-wrap: nowrap; gap: 8px; }}
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
"""


@dataclass
class Session:
    image_bgr: np.ndarray
    display_bgr: np.ndarray
    scale: float
    taps: list[tuple[int, int]] = field(default_factory=list)


def load_app_models() -> Models:
    models = load_models()
    if models is None:
        raise RuntimeError("models are missing from models/, promote them with python -m pipeline.promote")
    if torch.cuda.is_available():
        for model in (models.pieces, models.colors, models.corners, models.tiles, models.tokens):
            model.to("cuda")
    return models


MODELS = load_app_models()
BOARD = build_board()


@spaces.GPU(duration=30)
def analyze(image_bgr: np.ndarray, corners: list[list[float]] | None = None) -> BoardResult:
    return read_board(image_bgr, MODELS, BOARD, corners=corners)


def make_session(image_rgb: np.ndarray) -> Session:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    scale = min(1.0, DISPLAY_MAX / max(image_bgr.shape[:2]))
    display = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image_bgr
    return Session(image_bgr, display, scale)


def run(session: Session, corners: list[list[float]] | None) -> tuple[BoardResult, float]:
    start = time.perf_counter()
    try:
        result = analyze(session.image_bgr, corners)
    except Exception as error:
        logger.exception("reading the board failed")
        raise gr.Error(f"Couldn't read this photo: {error}") from error
    seconds = time.perf_counter() - start
    h, w = session.image_bgr.shape[:2]
    logger.info("read %dx%d photo in %.2f s, corners=%s, problem=%s", w, h, seconds,
                "manual" if corners else "auto", result.problem)
    return result, seconds


def data_url(image_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise gr.Error("Couldn't encode the image for display.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def stage_html(session: Session, result: BoardResult) -> str:
    board_only = result.rendered[:, :CANVAS]
    return (f'<div class="stage"><img class="photo" src="{data_url(session.display_bgr)}" alt="board photo">'
            f'<div class="board-layer"><img src="{data_url(board_only)}" alt="regenerated board"></div></div>')


def css_color(color: str) -> str:
    b, g, r = PLAYER_BGR.get(color, (128, 128, 128))
    return f"rgb({r}, {g}, {b})"


def result_info(result: BoardResult, seconds: float) -> str:
    players = "".join(
        f'<span><span class="dot" style="background:{css_color(s.color)}"></span>{s.color} {s.points}'
        f'{" (Longest Road)" if s.has_longest_road else ""}</span>'
        for s in result.scores
    ) or "<span>No pieces found on the board</span>"
    notes = " · ".join(result.warnings + [f"read in {seconds:.1f} s"])
    return f'<div class="scores">{players}</div><div class="notes">{notes}</div>'


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


def upload_view() -> tuple:
    return (gr.update(value=None, visible=True), gr.update(value=None, visible=False), gr.update(value="", visible=False),
            "", None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False))


def tap_view(session: Session, message: str = "") -> tuple:
    return (gr.update(visible=False), gr.update(value=tap_image(session), visible=True),
            gr.update(value="", visible=False), tap_info(session, message), session,
            gr.update(visible=True, interactive=bool(session.taps)),
            gr.update(visible=True, interactive=len(session.taps) == NUM_CORNERS), gr.update(visible=False),
            gr.update(visible=True), gr.update(visible=True))


def result_view(session: Session, result: BoardResult, seconds: float) -> tuple:
    return (gr.update(visible=False), gr.update(value=None, visible=False),
            gr.update(value=stage_html(session, result), visible=True), result_info(result, seconds), session,
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True),
            gr.update(visible=False))


def on_upload_photo(image_rgb: np.ndarray | None) -> tuple:
    if image_rgb is None:
        return upload_view()
    session = make_session(image_rgb)
    result, seconds = run(session, None)
    if result.problem:
        return tap_view(session, f"Couldn't find the corners automatically ({result.problem}).")
    return result_view(session, result, seconds)


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
    result, seconds = run(session, ordered.tolist())
    if result.problem:
        return tap_view(session, f"Couldn't read the board with these corners ({result.problem}).")
    return result_view(session, result, seconds)


def on_click_fix_corners(session: Session | None) -> tuple:
    if session is None:
        return upload_view()
    session.taps = []
    return tap_view(session)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Catan Board Reader") as demo:
        gr.Markdown("# Catan Board Reader", elem_id="title")
        state = gr.State(None)
        photo = gr.Image(type="numpy", sources=["upload"], image_mode="RGB", show_label=False, buttons=[],
                         elem_id="photo")
        taps = gr.Image(type="numpy", interactive=False, show_label=False, buttons=[], visible=False, elem_id="taps")
        stage = gr.HTML(elem_id="stage", visible=False, padding=False)
        info = gr.HTML("Take a photo of the whole board from above.", elem_id="info")
        with gr.Row(elem_id="actions"):
            undo_button = gr.Button("Undo", size="sm", visible=False, min_width=0)
            corners_button = gr.Button("Read board", variant="primary", size="sm", visible=False, min_width=0)
            fix_button = gr.Button("Fix corners", size="sm", visible=False, min_width=0)
            new_button = gr.Button("New photo", size="sm", visible=False, min_width=0)
        guide = gr.Image(GUIDE, interactive=False, show_label=False, buttons=[], visible=False, elem_id="guide")

        outputs = [photo, taps, stage, info, state, undo_button, corners_button, fix_button, new_button, guide]
        photo.upload(on_upload_photo, inputs=photo, outputs=outputs, api_name="read")
        photo.clear(upload_view, outputs=outputs, api_name=False)
        taps.select(on_select_tap, inputs=state, outputs=outputs, api_name=False)
        undo_button.click(on_click_undo, inputs=state, outputs=outputs, api_name=False)
        corners_button.click(on_click_read_corners, inputs=state, outputs=outputs, api_name="read_corners")
        fix_button.click(on_click_fix_corners, inputs=state, outputs=outputs, api_name=False)
        new_button.click(upload_view, outputs=outputs, api_name=False)
    return demo


if __name__ == "__main__":
    build_ui().launch(share="--share" in sys.argv, css=CSS)
