import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

from ultralytics import YOLO

from pipeline.corners import CORNER_IMGSZ
from pipeline.hexes import TILE_IMGSZ, TOKEN_IMGSZ
from pipeline.snap import COLOR_IMGSZ, IMGSZ
from pipeline.weights import MANIFEST, MODEL_DIR, MODEL_NAMES, weights_path

EXPECTED = {
    "pieces": ("detect", IMGSZ, {"city", "road", "robber", "settlement"}),
    "colors": ("classify", COLOR_IMGSZ, {"blue", "orange", "red", "white"}),
    "corners": ("detect", CORNER_IMGSZ, {"corner"}),
    "tiles": ("classify", TILE_IMGSZ, {"brick", "desert", "ore", "sheep", "wheat", "wood"}),
    "tokens": ("classify", TOKEN_IMGSZ, {"2", "3", "4", "5", "6", "8", "9", "10", "11", "12", "none"}),
}


def load_manifest() -> dict[str, dict[str, object]]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name: str, model: YOLO) -> list[str]:
    task, imgsz, names = EXPECTED[name]
    problems = []
    if model.task != task:
        problems.append(f"task is {model.task}, {name} needs {task}")
    trained_imgsz = model.ckpt.get("train_args", {}).get("imgsz")
    if trained_imgsz != imgsz:
        problems.append(f"trained at imgsz {trained_imgsz}, the pipeline runs {name} at {imgsz}")
    found = set(model.names.values())
    if found != names:
        problems.append(f"classes {sorted(found)} do not match {sorted(names)}")
    return problems


def promote(name: str, source: Path) -> bool:
    if not source.is_file():
        print(f"no file at {source}")
        return False
    model = YOLO(str(source))
    problems = check(name, model)
    if problems:
        print(f"not promoting {source} as {name}:")
        for problem in problems:
            print(f"  {problem}")
        return False

    target = weights_path(name)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    manifest = load_manifest()
    manifest[name] = {
        "source": source.as_posix(),
        "trained": model.ckpt.get("date"),
        "promoted": date.today().isoformat(),
        "task": model.task,
        "imgsz": EXPECTED[name][1],
        "sha256": file_hash(target),
    }
    ordered = {key: manifest[key] for key in MODEL_NAMES if key in manifest}
    MANIFEST.write_text(json.dumps(ordered, indent=2) + "\n")
    print(f"promoted {source} -> {target}")
    return True


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in MODEL_NAMES:
        print(f"usage: python -m pipeline.promote {{{','.join(MODEL_NAMES)}}} path/to/best.pt")
        return
    promote(sys.argv[1], Path(sys.argv[2]))


if __name__ == "__main__":
    main()
