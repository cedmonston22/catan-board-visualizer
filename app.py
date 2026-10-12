import logging
import sys
import time

import cv2
import gradio as gr
import numpy as np
import spaces
import torch

from pipeline.board import build_board
from pipeline.read_board import BoardResult, Models, load_models, read_board

logger = logging.getLogger("catan_app")
logging.basicConfig(level=logging.INFO)

SCORE_HEADERS = ["Player", "Points", "Settlements", "Cities", "Roads", "Longest road", "Longest Road bonus"]


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


def score_rows(result: BoardResult) -> list[list[object]]:
    return [[s.color, s.points, s.settlements, s.cities, s.roads, s.longest_road, "yes" if s.has_longest_road else ""]
            for s in result.scores]


def notes(result: BoardResult, seconds: float) -> str:
    lines = [f"- {w}" for w in result.warnings]
    lines.append(f"- read in {seconds:.1f} s")
    return "\n".join(lines)


def on_click_read(image_rgb: np.ndarray | None) -> tuple[np.ndarray | None, list[list[object]], str]:
    if image_rgb is None:
        raise gr.Error("Take or upload a photo of the board first.")
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    start = time.perf_counter()
    try:
        result = analyze(image_bgr)
    except Exception as error:
        logger.exception("reading the board failed")
        raise gr.Error(f"Couldn't read this photo: {error}") from error
    seconds = time.perf_counter() - start
    logger.info("read %dx%d photo in %.2f s, problem=%s", image_rgb.shape[1], image_rgb.shape[0], seconds, result.problem)

    if result.problem:
        return None, [], f"**Couldn't find the board's corners automatically** ({result.problem}). Try another photo with the whole board in view."
    rendered = cv2.cvtColor(result.rendered, cv2.COLOR_BGR2RGB)
    return rendered, score_rows(result), notes(result, seconds)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Catan Board Reader") as demo:
        gr.Markdown("# Catan Board Reader\nTake a photo of the whole board from above, then read it.")
        photo = gr.Image(label="Board photo", type="numpy", sources=["upload"], image_mode="RGB")
        read_button = gr.Button("Read board", variant="primary")
        board = gr.Image(label="Board", type="numpy", interactive=False)
        scores = gr.Dataframe(headers=SCORE_HEADERS, label="Scores", interactive=False)
        message = gr.Markdown()
        read_button.click(on_click_read, inputs=photo, outputs=[board, scores, message], api_name="read")
    return demo


if __name__ == "__main__":
    build_ui().launch(share="--share" in sys.argv)
