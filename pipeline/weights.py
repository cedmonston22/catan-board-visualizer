import json
from pathlib import Path

MODEL_DIR = Path("models")
MANIFEST = MODEL_DIR / "manifest.json"
MODEL_NAMES = ("pieces", "colors", "corners", "tiles", "tokens")


def weights_path(name: str) -> Path:
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}, expected one of {MODEL_NAMES}")
    return MODEL_DIR / f"{name}.pt"


def promoted_from(name: str) -> str:
    if MANIFEST.exists():
        entry = json.loads(MANIFEST.read_text()).get(name)
        if entry:
            return str(entry["source"])
    return weights_path(name).as_posix()
